package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/url"

	"github.com/hekmon/httplog/v3"
)

// TokenizeMessagesRequest represents the request body for /tokenize endpoint
type TokenizeMessagesRequest struct {
	Messages            []any          `json:"messages"`
	AddGenerationPrompt bool           `json:"add_generation_prompt,omitempty"`
	ReturnTokenStrings  bool           `json:"return_token_strs,omitempty"`
	ChatTemplateKwargs  map[string]any `json:"chat_template_kwargs,omitempty"`
	Tools               []any          `json:"tools,omitempty"`
}

// tokenize handles the /tokenize endpoint by intercepting requests and converting
// Responses API format messages to Chat Completions format if needed before forwarding to vLLM
func tokenize(httpCli *http.Client, target *url.URL,
	servedModel, thinkingGeneral, thinkingCoding, instructGeneral, instructReasoning string) http.HandlerFunc {

	return func(w http.ResponseWriter, r *http.Request) {
		// Prepare
		logger := logger.With(httplog.GetReqIDSLogAttr(r.Context()))
		ctx := r.Context()

		// Read request body
		requestBody, err := io.ReadAll(r.Body)
		if err != nil {
			logger.Error("failed to read body", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}

		// Parse request body into typed struct
		var req TokenizeMessagesRequest
		err = json.Unmarshal(requestBody, &req)
		if err != nil {
			logger.Error("failed to parse body as JSON", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusBadRequest)
			return
		}

		// Validate messages is present
		if len(req.Messages) == 0 {
			logger.Error("missing messages in request body")
			httpError(ctx, w, http.StatusBadRequest)
			return
		}

		// Check if messages need conversion from Responses to Chat Completions format
		messages := req.Messages
		needsConversion := false
		if len(messages) > 0 {
			messages, needsConversion = checkMessagesFormat(messages, logger)
		}
		if needsConversion {
			logger.Info("converting messages from Responses to Chat Completions format")
			// Convert messages using the same logic as responses.go
			convertedMessages, err := convertMessagesToChatFormat(messages, map[string]any{"instructions": ""}, logger)
			if err != nil {
				logger.Error("failed to convert messages", slog.Any("error", err))
				httpError(ctx, w, http.StatusBadRequest)
				return
			}
			req.Messages = convertedMessages

			// Convert tools from Responses format to Chat Completions format
			if len(req.Tools) > 0 {
				chatTools := make([]map[string]any, 0, len(req.Tools))
				for _, tool := range req.Tools {
					toolMap, ok := tool.(map[string]any)
					if !ok {
						continue
					}
					toolType, _ := toolMap["type"].(string)
					if toolType == "function" {
						// Responses: {type, name, description, parameters, strict}
						// Chat Completions: {type: "function", function: {name, description, parameters, strict}}
						funcDef := map[string]any{}
						if name, ok := toolMap["name"].(string); ok {
							funcDef["name"] = name
						}
						if desc, ok := toolMap["description"].(string); ok {
							funcDef["description"] = desc
						}
						if params, ok := toolMap["parameters"]; ok {
							funcDef["parameters"] = params
						}
						if strict, ok := toolMap["strict"]; ok {
							funcDef["strict"] = strict
						}
						chatTools = append(chatTools, map[string]any{
							"type":     "function",
							"function": funcDef,
						})
					}
					// Other tool types (web_search, file_search, etc.) are not supported by Chat Completions
				}
				if len(chatTools) > 0 {
					// Convert []map[string]any to []any for assignment
					toolsAsAny := make([]any, len(chatTools))
					for i, t := range chatTools {
						toolsAsAny[i] = t
					}
					req.Tools = toolsAsAny
				}
			}
		}

		// Build request to forward to vLLM
		forwardReq := map[string]any{
			"model":    servedModel,
			"messages": req.Messages,
		}
		if req.AddGenerationPrompt {
			forwardReq["add_generation_prompt"] = req.AddGenerationPrompt
		}
		if req.ReturnTokenStrings {
			forwardReq["return_token_strs"] = req.ReturnTokenStrings
		}
		if len(req.ChatTemplateKwargs) > 0 {
			forwardReq["chat_template_kwargs"] = req.ChatTemplateKwargs
		}
		if len(req.Tools) > 0 {
			forwardReq["tools"] = req.Tools
		}

		// Marshal request body
		requestBody, err = json.Marshal(forwardReq)
		if err != nil {
			logger.Error("failed to marshal request body", slog.Any("error", err))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}

		logger.Debug("forwarding tokenize request to backend", slog.String("body", string(requestBody)))

		// Track modified requests
		modifiedRequests.Add(1)

		// Prepare outgoing request
		outreq := r.Clone(ctx)
		rewriteRequestURL(outreq, target)
		outreq.Body = io.NopCloser(bytes.NewReader(requestBody))
		outreq.ContentLength = int64(len(requestBody))
		outreq.RequestURI = ""

		// Send request to backend
		outResp, err := httpCli.Do(outreq)
		if err != nil {
			logger.Error("failed to send upstream request", slog.Any("error", err))
			httpError(ctx, w, http.StatusBadGateway)
			return
		}
		defer outResp.Body.Close()

		// Copy response headers
		for header, values := range outResp.Header {
			for _, value := range values {
				w.Header().Add(header, value)
			}
		}

		// Write response
		w.WriteHeader(outResp.StatusCode)
		if _, err = io.Copy(w, outResp.Body); err != nil {
			logger.Error("failed to write response", slog.String("error", err.Error()))
		}
	}
}

// checkMessagesFormat checks if messages are in Responses API format or Chat Completions format
// Returns the messages array and a boolean indicating if conversion is needed
func checkMessagesFormat(messagesAny any, logger *slog.Logger) ([]any, bool) {
	messages, ok := messagesAny.([]any)
	if !ok {
		logger.Warn("messages is not an array")
		return nil, false
	}

	if len(messages) == 0 {
		return messages, false
	}

	// Check the first message to determine format
	// Chat Completions format: {role: "user"|"assistant"|"system"|"tool", content: ...}
	// Responses format: {type: "message"|"function_call_output", role: ..., content: ...}
	firstMsg, ok := messages[0].(map[string]any)
	if !ok {
		return messages, false
	}

	// If message has "type" field, it's likely Responses API format
	if _, hasType := firstMsg["type"]; hasType {
		logger.Debug("detected Responses API format messages (has 'type' field)")
		return messages, true
	}

	// Check for Responses API specific fields
	if _, hasCallID := firstMsg["call_id"]; hasCallID {
		logger.Debug("detected Responses API format messages (has 'call_id' field)")
		return messages, true
	}

	logger.Debug("detected Chat Completions format messages")
	return messages, false
}

// convertMessagesToChatFormat converts messages from Responses API format to Chat Completions format
func convertMessagesToChatFormat(messages []any, reqData map[string]any, logger *slog.Logger) ([]any, error) {
	var chatMessages []any

	// Check if instructions field exists (Responses API)
	if instr, ok := reqData["instructions"].(string); ok && instr != "" {
		chatMessages = append(chatMessages, map[string]any{
			"role":    "system",
			"content": instr,
		})
	}

	for _, msgAny := range messages {
		msgMap, ok := msgAny.(map[string]any)
		if !ok {
			continue
		}

		msgType, _ := msgMap["type"].(string)

		switch msgType {
		case "message", "":
			// Regular message in Responses format
			role, _ := msgMap["role"].(string)
			if role == "" {
				role = "user"
			}
			content := msgMap["content"]

			// Convert content if it's an array of parts
			if contentArray, ok := content.([]any); ok {
				content = convertContentPartsToChatFormat(contentArray, logger)
			}

			chatMessages = append(chatMessages, map[string]any{
				"role":    role,
				"content": content,
			})

		case "function_call_output":
			// Tool call result
			callID, _ := msgMap["call_id"].(string)
			output, _ := msgMap["output"].(string)

			chatMessages = append(chatMessages, map[string]any{
				"role":         "tool",
				"tool_call_id": callID,
				"content":      output,
			})

		default:
			// Pass through other message types as-is
			chatMessages = append(chatMessages, msgMap)
		}
	}

	return chatMessages, nil
}
