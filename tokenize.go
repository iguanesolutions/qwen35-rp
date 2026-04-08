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

// tokenize handles the /tokenize endpoint by intercepting requests and converting
// Responses API format messages to Chat Completions format if needed before forwarding to vLLM.
//
// Accepted input formats:
//   - Chat Completions: {"messages": [...], "tools": [...]}
//   - Responses API:    {"input": "..." or [...], "instructions": "...", "tools": [...]}
//   - vLLM prompt:      {"prompt": "..."}
func tokenize(httpCli *http.Client, target *url.URL, servedModel string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		logger := logger.With(httplog.GetReqIDSLogAttr(r.Context()))
		ctx := r.Context()

		// Read request body
		r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodySize)
		requestBody, err := io.ReadAll(r.Body)
		if err != nil {
			logger.Error("failed to read body", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}

		// Parse as generic map to detect format
		var reqData map[string]any
		if err := json.Unmarshal(requestBody, &reqData); err != nil {
			logger.Error("failed to parse body as JSON", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusBadRequest)
			return
		}

		// Determine the messages and tools in Chat Completions format.
		// Exactly one conversion path is taken depending on which fields are present.
		var (
			messages []map[string]any
			tools    []map[string]any
			modified bool
		)

		switch {
		case reqData["messages"] != nil:
			// Chat Completions format: messages are already in the right shape.
			// We still normalize content parts (e.g. input_text → text) for safety.
			logger.Debug("detected Chat Completions format in tokenize request")
			rawMessages, ok := reqData["messages"].([]any)
			if !ok || len(rawMessages) == 0 {
				logger.Error("messages field is not a valid array")
				httpError(ctx, w, http.StatusBadRequest)
				return
			}
			messages = normalizeChatMessages(rawMessages, logger)
			// Tools are already in Chat Completions format, pass through as-is
			if rawTools, ok := reqData["tools"].([]any); ok && len(rawTools) > 0 {
				tools = passThroughTools(rawTools)
			}

		case reqData["input"] != nil:
			// Responses API format: convert input → messages, tools → chat tools
			logger.Info("detected Responses API format in tokenize request")
			messages = convertInputToMessages(reqData["input"], reqData["instructions"], logger)
			if rawTools, ok := reqData["tools"].([]any); ok && len(rawTools) > 0 {
				tools = convertToolsToChat(rawTools)
			}
			modified = true

		case reqData["prompt"] != nil:
			// vLLM native prompt format: wrap in a single user message
			promptStr, ok := reqData["prompt"].(string)
			if !ok || promptStr == "" {
				logger.Error("prompt field is not a valid string")
				httpError(ctx, w, http.StatusBadRequest)
				return
			}
			logger.Info("detected prompt tokenization request")
			messages = []map[string]any{
				{"role": "user", "content": promptStr},
			}
			modified = true

		default:
			logger.Error("request must contain 'messages', 'input', or 'prompt' field")
			httpError(ctx, w, http.StatusBadRequest)
			return
		}

		if len(messages) == 0 {
			logger.Error("no messages could be derived from request")
			httpError(ctx, w, http.StatusBadRequest)
			return
		}

		// Build the forward request for vLLM
		forwardReq := map[string]any{
			"model":    servedModel,
			"messages": messages,
		}
		// Carry over tokenize-specific options
		if v, ok := reqData["add_generation_prompt"]; ok {
			forwardReq["add_generation_prompt"] = v
		}
		if v, ok := reqData["return_token_strs"]; ok {
			forwardReq["return_token_strs"] = v
		}
		if v, ok := reqData["chat_template_kwargs"]; ok {
			forwardReq["chat_template_kwargs"] = v
		}
		if len(tools) > 0 {
			forwardReq["tools"] = tools
		}

		// Marshal request body
		requestBody, err = json.Marshal(forwardReq)
		if err != nil {
			logger.Error("failed to marshal request body", slog.Any("error", err))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}

		logger.Debug("forwarding tokenize request to backend", slog.String("body", string(requestBody)))

		if modified {
			modifiedRequests.Add(1)
		}

		// Prepare outgoing request
		outreq := r.Clone(ctx)
		rewriteRequestURL(outreq, target)
		stripHopByHopHeaders(outreq)
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

		// Copy response headers and body
		for header, values := range outResp.Header {
			for _, value := range values {
				w.Header().Add(header, value)
			}
		}
		w.WriteHeader(outResp.StatusCode)
		if _, err = io.Copy(w, outResp.Body); err != nil {
			logger.Error("failed to write response", slog.String("error", err.Error()))
		}
	}
}

// normalizeChatMessages processes Chat Completions messages, normalizing content parts
// (e.g. converting Responses API content part types like input_text to Chat format).
// This handles the case where a caller sends messages in Chat Completions structure
// but with mixed content part formats.
func normalizeChatMessages(rawMessages []any, logger *slog.Logger) []map[string]any {
	messages := make([]map[string]any, 0, len(rawMessages))
	for _, msgAny := range rawMessages {
		msgMap, ok := msgAny.(map[string]any)
		if !ok {
			continue
		}
		role, _ := msgMap["role"].(string)
		content := msgMap["content"]

		// Normalize content parts if present as array
		if contentArray, ok := content.([]any); ok {
			content = convertContentPartsToChatFormat(contentArray, logger)
		}

		msg := map[string]any{
			"role":    role,
			"content": content,
		}
		// Preserve tool_call_id for tool role messages
		if toolCallID, ok := msgMap["tool_call_id"].(string); ok {
			msg["tool_call_id"] = toolCallID
		}
		// Preserve tool_calls for assistant messages
		if toolCalls, ok := msgMap["tool_calls"]; ok {
			msg["tool_calls"] = toolCalls
		}
		messages = append(messages, msg)
	}
	return messages
}

// passThroughTools converts []any to []map[string]any, passing tools through as-is.
// Used for Chat Completions format tools that are already in the correct structure.
func passThroughTools(rawTools []any) []map[string]any {
	tools := make([]map[string]any, 0, len(rawTools))
	for _, t := range rawTools {
		if toolMap, ok := t.(map[string]any); ok {
			tools = append(tools, toolMap)
		}
	}
	return tools
}
