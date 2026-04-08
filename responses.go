package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/hekmon/httplog/v3"
)

// responses handles the OpenAI Responses API endpoint by converting requests to Chat Completions
// format and responses back to Responses format.
//
// Why convert to Chat Completions instead of forwarding to vLLM's /v1/responses endpoint?
// vLLM's Responses endpoint does not support chat_template_kwargs, which is required to control
// Qwen's thinking mode (enable_thinking=true/false). By converting to Chat Completions, we can
// properly set this parameter and control whether the model uses thinking/reasoning mode.
func responses(httpCli *http.Client, target *url.URL,
	servedModel, thinkingGeneral, thinkingCoding, instructGeneral, instructReasoning string, enforceSamplingParams bool) http.HandlerFunc {

	return func(w http.ResponseWriter, r *http.Request) {
		// Prepare
		logger := logger.With(httplog.GetReqIDSLogAttr(r.Context()))
		ctx := r.Context()
		var think, stream bool

		// Read request body
		r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodySize)
		requestBody, err := io.ReadAll(r.Body)
		if err != nil {
			logger.Error("failed to read body", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}

		// Parse request body
		var reqData map[string]any
		err = json.Unmarshal(requestBody, &reqData)
		if err != nil {
			logger.Error("failed to parse body as JSON", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusBadRequest)
			return
		}

		// Extract and validate model
		modelName, ok := reqData["model"].(string)
		if !ok || modelName == "" {
			logger.Error("missing/invalid model in request body")
			httpError(ctx, w, http.StatusBadRequest)
			return
		}

		// Validate input is present (required field per OpenAI spec)
		if reqData["input"] == nil {
			logger.Error("missing input in request body")
			httpError(ctx, w, http.StatusBadRequest)
			return
		}

		// Track streaming mode
		if streamVal, ok := reqData["stream"]; ok {
			stream, _ = streamVal.(bool)
		}

		// Validate model and apply profiles
		var samplingParams map[string]any
		switch modelName {
		case thinkingGeneral:
			think = true
			samplingParams = thinkingGeneralParams
			logger.Info("model matched", slog.String("type", "thinking_general"), slog.String("virtual_model", modelName))
		case thinkingCoding:
			think = true
			samplingParams = thinkingCodingParams
			logger.Info("model matched", slog.String("type", "thinking_coding"), slog.String("virtual_model", modelName))
		case instructGeneral:
			think = false
			samplingParams = instructGeneralParams
			logger.Info("model matched", slog.String("type", "instruct_general"), slog.String("virtual_model", modelName))
		case instructReasoning:
			think = false
			samplingParams = instructReasoningParams
			logger.Info("model matched", slog.String("type", "instruct_reasoning"), slog.String("virtual_model", modelName))
		default:
			logger.Error("unsupported model", slog.String("model", modelName))
			httpError(ctx, w, http.StatusBadRequest)
			return
		}

		// Convert Responses request to Chat Completions format
		chatData, err := convertResponsesToChat(reqData, logger)
		if err != nil {
			logger.Error("failed to convert Responses to Chat format", slog.Any("error", err))
			httpError(ctx, w, http.StatusBadRequest)
			return
		}

		// Apply sampling parameters
		applySamplingParams(chatData, samplingParams, logger, enforceSamplingParams)

		// Set thinking mode via chat_template_kwargs
		kwargs, ok := chatData["chat_template_kwargs"]
		if ok {
			kwargsMap, ok := kwargs.(map[string]any)
			if !ok {
				logger.Error("chat_template_kwargs is not a map[string]any")
				httpError(ctx, w, http.StatusBadRequest)
				return
			}
			kwargsMap["enable_thinking"] = think
			chatData["chat_template_kwargs"] = kwargsMap
		} else {
			chatData["chat_template_kwargs"] = map[string]any{"enable_thinking": think}
		}

		// Override model name for backend
		chatData["model"] = servedModel

		// Marshal chat request
		requestBody, err = json.Marshal(chatData)
		if err != nil {
			logger.Error("failed to marshal chat request body", slog.Any("error", err))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}

		logger.Debug("rewritten request body", slog.String("body", string(requestBody)))
		modifiedRequests.Add(1)

		// Prepare outgoing request to /v1/chat/completions
		outreq := r.Clone(ctx)
		rewriteRequestURL(outreq, target)
		stripHopByHopHeaders(outreq)
		outreq.URL.Path = "/v1/chat/completions"
		outreq.Body = io.NopCloser(bytes.NewReader(requestBody))
		outreq.ContentLength = int64(len(requestBody))
		outreq.RequestURI = ""

		// Send request
		outResp, err := httpCli.Do(outreq)
		if err != nil {
			logger.Error("failed to send upstream request", slog.Any("error", err))
			switch {
			case errors.Is(err, syscall.ECONNREFUSED):
				httpError(ctx, w, http.StatusBadGateway)
			default:
				httpError(ctx, w, http.StatusInternalServerError)
			}
			return
		}
		defer outResp.Body.Close()

		if stream {
			// Streaming mode: copy headers and convert Chat SSE to Responses SSE
			logger.Debug("streaming response to client with Responses format")
			copyHeaders(w, outResp)
			w.WriteHeader(outResp.StatusCode)
			if err = streamResponsesConverter(w, outResp.Body, modelName, logger); err != nil {
				logger.Error("failed to stream response", slog.String("error", err.Error()))
			}
		} else {
			// Non-streaming mode: read full response and convert
			responseBody, err := io.ReadAll(outResp.Body)
			if err != nil {
				logger.Error("failed to read response body", slog.String("error", err.Error()))
				httpError(ctx, w, http.StatusInternalServerError)
				return
			}

			// Fix vLLM bug (non-thinking responses with content in reasoning_content field)
			// before converting to Responses format
			responseBody = fixNonStreamingResponse(responseBody, think, modelName, logger)

			// Parse and convert response
			var chatResp map[string]any
			if err := json.Unmarshal(responseBody, &chatResp); err != nil {
				logger.Error("failed to parse response JSON", slog.Any("error", err))
				httpError(ctx, w, http.StatusInternalServerError)
				return
			}

			// Convert to Responses format
			respData, err := convertChatToResponses(chatResp, modelName, logger)
			if err != nil {
				logger.Error("failed to convert response to Responses format", slog.Any("error", err))
				httpError(ctx, w, http.StatusInternalServerError)
				return
			}

			// Write response
			responseBody, err = json.Marshal(respData)
			if err != nil {
				logger.Error("failed to marshal Responses response", slog.Any("error", err))
				httpError(ctx, w, http.StatusInternalServerError)
				return
			}

			copyHeaders(w, outResp)
			w.Header().Set("Content-Length", strconv.Itoa(len(responseBody)))
			w.WriteHeader(outResp.StatusCode)
			if _, err = w.Write(responseBody); err != nil {
				logger.Error("failed to write response", slog.String("error", err.Error()))
			}
		}
	}
}

// convertResponsesToChat converts Responses API request to Chat Completions format
func convertResponsesToChat(reqData map[string]any, logger *slog.Logger) (map[string]any, error) {
	chatData := make(map[string]any)

	// Copy model (will be overridden later)
	if model, ok := reqData["model"]; ok {
		chatData["model"] = model
	}

	// Convert input → messages
	chatData["messages"] = convertInputToMessages(reqData["input"], reqData["instructions"], logger)

	// Copy sampling parameters
	if temp, ok := reqData["temperature"]; ok {
		chatData["temperature"] = temp
	}
	if topP, ok := reqData["top_p"]; ok {
		chatData["top_p"] = topP
	}
	if maxTokens, ok := reqData["max_output_tokens"]; ok {
		chatData["max_tokens"] = maxTokens
	}

	// Convert tools from Responses format to Chat Completions format
	if tools, ok := reqData["tools"].([]any); ok && len(tools) > 0 {
		if chatTools := convertToolsToChat(tools); len(chatTools) > 0 {
			chatData["tools"] = chatTools
		}
	}

	// Copy tool_choice if present
	if toolChoice, ok := reqData["tool_choice"]; ok {
		chatData["tool_choice"] = toolChoice
	}

	// Copy response_format if present
	if respFormat, ok := reqData["response_format"]; ok {
		chatData["response_format"] = respFormat
	}

	// Copy stream flag and add stream_options for usage tracking
	if stream, ok := reqData["stream"]; ok {
		chatData["stream"] = stream
		// Always request usage in streaming mode - critical for billing
		if streamBool, ok := stream.(bool); ok && streamBool {
			chatData["stream_options"] = map[string]any{
				"include_usage": true,
			}
		}
	}

	return chatData, nil
}

// convertInputToMessages converts Responses input to Chat Completions messages array
func convertInputToMessages(input any, instructions any, logger *slog.Logger) []map[string]any {
	var messages []map[string]any

	// Add system instruction if present
	if instructions != nil {
		if instrStr, ok := instructions.(string); ok && instrStr != "" {
			messages = append(messages, map[string]any{
				"role":    "system",
				"content": instrStr,
			})
		}
	}

	switch v := input.(type) {
	case string:
		// Simple string input → single user message
		messages = append(messages, map[string]any{
			"role":    "user",
			"content": v,
		})

	case []any:
		// Array of input items.
		// We need to group consecutive function_call items into a single assistant message
		// with a tool_calls array, as Chat Completions format requires.
		var pendingToolCalls []map[string]any

		// flushToolCalls appends a single assistant message containing all pending tool calls
		flushToolCalls := func() {
			if len(pendingToolCalls) == 0 {
				return
			}
			messages = append(messages, map[string]any{
				"role":       "assistant",
				"content":    nil,
				"tool_calls": pendingToolCalls,
			})
			pendingToolCalls = nil
		}

		for _, item := range v {
			itemMap, ok := item.(map[string]any)
			if !ok {
				continue
			}

			itemType, _ := itemMap["type"].(string)

			switch itemType {
			case "message", "":
				flushToolCalls()
				// Regular message
				role, _ := itemMap["role"].(string)
				if role == "" {
					role = "user"
				}
				content := itemMap["content"]

				// Convert content if it's an array of parts
				if contentArray, ok := content.([]any); ok {
					content = convertContentPartsToChatFormat(contentArray, logger)
				}

				messages = append(messages, map[string]any{
					"role":    role,
					"content": content,
				})

			case "function_call":
				// Assistant tool call — accumulate into pending group
				callID, _ := itemMap["call_id"].(string)
				name, _ := itemMap["name"].(string)
				args, _ := itemMap["arguments"].(string)

				pendingToolCalls = append(pendingToolCalls, map[string]any{
					"id":   callID,
					"type": "function",
					"function": map[string]any{
						"name":      name,
						"arguments": args,
					},
				})

			case "function_call_output":
				flushToolCalls()
				// Tool call result
				callID, _ := itemMap["call_id"].(string)
				output, _ := itemMap["output"].(string)

				messages = append(messages, map[string]any{
					"role":         "tool",
					"tool_call_id": callID,
					"content":      output,
				})
			}
		}
		flushToolCalls()
	}

	return messages
}

// convertToolsToChat converts tools from Responses API format to Chat Completions format.
// Responses: {type: "function", name, description, parameters, strict}
// Chat Completions: {type: "function", function: {name, description, parameters, strict}}
func convertToolsToChat(tools []any) []map[string]any {
	chatTools := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		toolMap, ok := tool.(map[string]any)
		if !ok {
			continue
		}
		toolType, _ := toolMap["type"].(string)
		if toolType != "function" {
			// Other tool types (web_search, file_search, etc.) are not supported by Chat Completions
			continue
		}
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
	return chatTools
}

// convertContentPartsToChatFormat converts content parts from various formats to Chat Completions format
// Handles: Responses API (input_text, input_image), Chat Completions (text, image_url), and vLLM native formats
func convertContentPartsToChatFormat(parts []any, logger *slog.Logger) []map[string]any {
	var chatParts []map[string]any

	for _, part := range parts {
		partMap, ok := part.(map[string]any)
		if !ok {
			continue
		}

		partType, _ := partMap["type"].(string)

		switch partType {
		// Responses API format
		case "input_text":
			text, _ := partMap["text"].(string)
			chatParts = append(chatParts, map[string]any{
				"type": "text",
				"text": text,
			})

		case "input_image":
			imageURL, _ := partMap["image_url"].(string)
			detail, _ := partMap["detail"].(string)

			chatPart := map[string]any{
				"type": "image_url",
				"image_url": map[string]any{
					"url": imageURL,
				},
			}
			if detail != "" {
				chatPart["image_url"].(map[string]any)["detail"] = detail
			}
			chatParts = append(chatParts, chatPart)

		// Chat Completions / vLLM native format - ensure correct structure
		case "text":
			text, _ := partMap["text"].(string)
			chatParts = append(chatParts, map[string]any{
				"type": "text",
				"text": text,
			})

		case "image_url":
			// Handle both formats: {"image_url": "string"} and {"image_url": {"url": "string"}}
			imageURLData := partMap["image_url"]
			var imageURL string
			var detail string

			if urlMap, ok := imageURLData.(map[string]any); ok {
				// Nested format: {"image_url": {"url": "...", "detail": "..."}}
				if url, ok := urlMap["url"].(string); ok {
					imageURL = url
				}
				if d, ok := urlMap["detail"].(string); ok {
					detail = d
				}
			} else if url, ok := imageURLData.(string); ok {
				// Flat format: {"image_url": "..."}
				imageURL = url
			}

			chatPart := map[string]any{
				"type": "image_url",
				"image_url": map[string]any{
					"url": imageURL,
				},
			}
			if detail != "" {
				chatPart["image_url"].(map[string]any)["detail"] = detail
			}
			chatParts = append(chatParts, chatPart)

		default:
			// Pass through unknown types as-is, but ensure structure is correct for vLLM
			chatParts = append(chatParts, partMap)
		}
	}

	return chatParts
}

// convertChatToResponses converts Chat Completions response to Responses API format
func convertChatToResponses(chatData map[string]any, virtualModel string, logger *slog.Logger) (map[string]any, error) {
	now := time.Now().Unix()

	// Convert choices to output items
	choices, ok := chatData["choices"].([]any)
	if !ok || len(choices) == 0 {
		// Return minimal response with no output
		respData := map[string]any{
			"id":                   fmt.Sprintf("resp_%s", generateSimpleID()),
			"created_at":           now,
			"incomplete_details":   nil,
			"instructions":         nil,
			"metadata":             nil,
			"model":                virtualModel,
			"object":               "response",
			"output":               []any{},
			"parallel_tool_calls":  true,
			"temperature":          nil,
			"tool_choice":          "auto",
			"tools":                []any{},
			"top_p":                nil,
			"background":           false,
			"max_output_tokens":    nil,
			"max_tool_calls":       nil,
			"previous_response_id": nil,
			"prompt":               nil,
			"reasoning":            nil,
			"service_tier":         "auto",
			"status":               "completed",
			"text":                 nil,
			"top_logprobs":         nil,
			"truncation":           "disabled",
			"usage":                convertUsage(map[string]any{}),
			"user":                 nil,
			"input_messages":       nil,
			"output_messages":      nil,
		}
		return respData, nil
	}

	var outputItems []any
	var totalText strings.Builder

	for _, choice := range choices {
		choiceMap, ok := choice.(map[string]any)
		if !ok {
			continue
		}

		message, ok := choiceMap["message"].(map[string]any)
		if !ok {
			continue
		}

		// Check for reasoning_content (thinking mode) - must come BEFORE message item
		if reasoning, ok := message["reasoning_content"].(string); ok && reasoning != "" {
			reasoningItem := map[string]any{
				"id":      fmt.Sprintf("rs_%s", generateSimpleID()),
				"summary": []any{},
				"type":    "reasoning",
				"content": []any{
					map[string]any{
						"text": reasoning,
						"type": "reasoning_text",
					},
				},
				"encrypted_content": nil,
				"status":            nil,
			}
			outputItems = append(outputItems, reasoningItem)
		}

		// Build message output item
		outputItem := map[string]any{
			"id":      fmt.Sprintf("msg_%s", generateSimpleID()),
			"content": []any{},
			"role":    "assistant",
			"status":  "completed",
			"type":    "message",
			"phase":   nil,
		}

		var contentParts []any

		// Convert main content
		content, _ := message["content"].(string)
		if content != "" {
			totalText.WriteString(content)
			contentParts = append(contentParts, map[string]any{
				"annotations": []any{},
				"text":        content,
				"type":        "output_text",
				"logprobs":    nil,
			})
		}

		// Check for tool calls - add as separate items AFTER message
		if toolCalls, ok := message["tool_calls"].([]any); ok {
			for _, tc := range toolCalls {
				tcMap, ok := tc.(map[string]any)
				if !ok {
					continue
				}

				function, ok := tcMap["function"].(map[string]any)
				if !ok {
					continue
				}

				name, _ := function["name"].(string)
				args, _ := function["arguments"].(string)
				callID, _ := tcMap["id"].(string)
				if callID == "" {
					callID = fmt.Sprintf("call_%s", generateSimpleID())
				}

				toolCallItem := map[string]any{
					"id":        fmt.Sprintf("fc_%s", generateSimpleID()),
					"type":      "function_call",
					"call_id":   callID,
					"name":      name,
					"arguments": args,
					"status":    "completed",
				}
				outputItems = append(outputItems, toolCallItem)
			}
		}

		// Add content parts if not empty
		if len(contentParts) > 0 {
			outputItem["content"] = contentParts
			outputItems = append(outputItems, outputItem)
		}
	}

	// Check if response was truncated (finish_reason == "length")
	finishReason, _ := choices[0].(map[string]any)["finish_reason"].(string)
	var incompleteDetails map[string]any
	status := "completed"
	if finishReason == "length" {
		status = "incomplete"
		incompleteDetails = map[string]any{
			"reason": "max_output_tokens",
		}
	}

	// Build final response with all fields matching OpenAI Responses API
	respData := map[string]any{
		"id":                   fmt.Sprintf("resp_%s", generateSimpleID()),
		"created_at":           now,
		"incomplete_details":   incompleteDetails,
		"instructions":         nil,
		"metadata":             nil,
		"model":                virtualModel,
		"object":               "response",
		"output":               outputItems,
		"parallel_tool_calls":  true,
		"temperature":          nil,
		"tool_choice":          "auto",
		"tools":                []any{},
		"top_p":                nil,
		"background":           false,
		"max_output_tokens":    nil,
		"max_tool_calls":       nil,
		"previous_response_id": nil,
		"prompt":               nil,
		"reasoning":            nil,
		"service_tier":         "auto",
		"status":               status,
		"text":                 nil,
		"top_logprobs":         nil,
		"truncation":           "disabled",
		"usage":                convertUsage(chatData["usage"]),
		"user":                 nil,
		"input_messages":       nil,
		"output_messages":      nil,
	}

	respData["output_text"] = totalText.String()

	return respData, nil
}

// convertUsage converts Chat Completions usage to Responses format
func convertUsage(usage any) map[string]any {
	respUsage := map[string]any{
		"input_tokens":  0,
		"output_tokens": 0,
		"total_tokens":  0,
		"input_tokens_details": map[string]any{
			"cached_tokens":          0,
			"input_tokens_per_turn":  []any{},
			"cached_tokens_per_turn": []any{},
		},
		"output_tokens_details": map[string]any{
			"reasoning_tokens":            0,
			"tool_output_tokens":          0,
			"output_tokens_per_turn":      []any{},
			"tool_output_tokens_per_turn": []any{},
		},
	}

	if usage == nil {
		return respUsage
	}

	usageMap, ok := usage.(map[string]any)
	if !ok {
		return respUsage
	}

	// Copy basic token counts (Chat Completions uses prompt_tokens/completion_tokens)
	if promptTokens, ok := usageMap["prompt_tokens"]; ok {
		respUsage["input_tokens"] = promptTokens
	}
	if completionTokens, ok := usageMap["completion_tokens"]; ok {
		respUsage["output_tokens"] = completionTokens
	}
	if totalTokens, ok := usageMap["total_tokens"]; ok {
		respUsage["total_tokens"] = totalTokens
	}

	// Handle input_tokens_details
	if inputDetails, ok := usageMap["input_tokens_details"].(map[string]any); ok {
		respDetails := map[string]any{
			"cached_tokens":          0,
			"input_tokens_per_turn":  []any{},
			"cached_tokens_per_turn": []any{},
		}
		if cached, ok := inputDetails["cached_tokens"]; ok {
			respDetails["cached_tokens"] = cached
		}
		if perTurn, ok := inputDetails["input_tokens_per_turn"]; ok {
			respDetails["input_tokens_per_turn"] = perTurn
		}
		if cachedPerTurn, ok := inputDetails["cached_tokens_per_turn"]; ok {
			respDetails["cached_tokens_per_turn"] = cachedPerTurn
		}
		respUsage["input_tokens_details"] = respDetails
	}

	// Handle output_tokens_details
	if outputDetails, ok := usageMap["output_tokens_details"].(map[string]any); ok {
		respDetails := map[string]any{
			"reasoning_tokens":            0,
			"tool_output_tokens":          0,
			"output_tokens_per_turn":      []any{},
			"tool_output_tokens_per_turn": []any{},
		}
		if reasoning, ok := outputDetails["reasoning_tokens"]; ok {
			respDetails["reasoning_tokens"] = reasoning
		}
		if toolOutput, ok := outputDetails["tool_output_tokens"]; ok {
			respDetails["tool_output_tokens"] = toolOutput
		}
		if perTurn, ok := outputDetails["output_tokens_per_turn"]; ok {
			respDetails["output_tokens_per_turn"] = perTurn
		}
		if toolPerTurn, ok := outputDetails["tool_output_tokens_per_turn"]; ok {
			respDetails["tool_output_tokens_per_turn"] = toolPerTurn
		}
		respUsage["output_tokens_details"] = respDetails
	}

	return respUsage
}

// toolCallState tracks the state of a streaming tool call
type toolCallState struct {
	ID        string
	Name      string
	Arguments strings.Builder
	ItemID    string
	Index     int
	Started   bool
}

// responsesStreamState holds all mutable state for a streaming Responses API conversion
type responsesStreamState struct {
	responseID            string
	itemID                string
	virtualModel          string
	outputIndex           int
	contentIndex          int
	hasReasoning          bool
	reasoningItemID       string
	reasoningContentIndex int
	currentText           strings.Builder
	reasoningText         strings.Builder
	seqNum                int
	lastUsage             map[string]any
	messageStarted        bool
	finishReason          string
	toolCalls             map[int]*toolCallState
	logger                *slog.Logger
}

// streamResponsesConverter converts Chat Completions SSE to Responses SSE
func streamResponsesConverter(w http.ResponseWriter, backendBody io.ReadCloser, virtualModel string, logger *slog.Logger) error {
	s := &responsesStreamState{
		responseID:   fmt.Sprintf("resp_%s", generateSimpleID()),
		itemID:       fmt.Sprintf("msg_%s", generateSimpleID()),
		virtualModel: virtualModel,
		toolCalls:    make(map[int]*toolCallState),
		logger:       logger,
	}

	now := time.Now().Unix()

	buf := make([]byte, 0, 4096)
	temp := make([]byte, 4096)

	// Send initial response.created event
	initialResp := buildInitialResponse(s.responseID, s.virtualModel, now, "in_progress")
	sendSSEEvent(w, map[string]any{
		"type":            "response.created",
		"response":        initialResp,
		"sequence_number": s.seqNum,
	}, s.logger)
	s.seqNum++

	sendSSEEvent(w, map[string]any{
		"type":            "response.in_progress",
		"response":        initialResp,
		"sequence_number": s.seqNum,
	}, s.logger)
	s.seqNum++

	for {
		n, err := backendBody.Read(temp)
		if n > 0 {
			buf = append(buf, temp[:n]...)
		}
		if err != nil && err != io.EOF {
			return err
		}

		// Process complete SSE events (including any received in the final EOF read)
		for {
			idx := bytes.Index(buf, []byte("\n\n"))
			if idx == -1 {
				break
			}

			event := buf[:idx+2]
			buf = buf[idx+2:]

			if bytes.HasPrefix(event, []byte("data: ")) {
				jsonPart := bytes.TrimSpace(event[6:])
				if len(jsonPart) == 0 || bytes.Equal(jsonPart, []byte("[DONE]")) {
					continue
				}

				var chatEvent map[string]any
				if jsonErr := json.Unmarshal(jsonPart, &chatEvent); jsonErr != nil {
					continue
				}

				// Capture usage and finish_reason from chunk
				if usage, ok := chatEvent["usage"].(map[string]any); ok {
					s.lastUsage = usage
				}
				if choices, ok := chatEvent["choices"].([]any); ok && len(choices) > 0 {
					if choice, ok := choices[0].(map[string]any); ok {
						if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
							s.finishReason = fr
						}
					}
				}

				// Convert Chat event to Responses events
				respEvents := s.convertChatSSEEventToResponses(chatEvent)

				for _, respEvent := range respEvents {
					eventJSON, jerr := json.Marshal(respEvent)
					if jerr != nil {
						logger.Error("failed to marshal streaming event", slog.Any("error", jerr))
						continue
					}
					if _, werr := fmt.Fprintf(w, "data: %s\n\n", eventJSON); werr != nil {
						return werr
					}
					if flusher, ok := w.(http.Flusher); ok {
						flusher.Flush()
					}
				}
			}
		}

		if err == io.EOF {
			// Send completion events before response.completed
			s.sendCompletionEvents(w)

			// Send final completed event
			finalResp := buildFinalResponse(s.responseID, s.itemID, s.reasoningItemID, s.virtualModel, now, s.currentText.String(), s.reasoningText.String(), s.lastUsage, s.finishReason, s.toolCalls)
			sendSSEEvent(w, map[string]any{
				"type":            "response.completed",
				"response":        finalResp,
				"sequence_number": s.seqNum,
			}, s.logger)

			return nil
		}
	}
}

// sendCompletionEvents sends the completion events before response.completed
func (s *responsesStreamState) sendCompletionEvents(w http.ResponseWriter) {
	text := s.currentText.String()
	reasoning := s.reasoningText.String()

	// Send reasoning completion events if we had reasoning
	if s.hasReasoning && s.reasoningItemID != "" {
		// response.reasoning_text.done
		sendSSEEvent(w, map[string]any{
			"type":            "response.reasoning_text.done",
			"item_id":         s.reasoningItemID,
			"output_index":    0,
			"content_index":   0,
			"text":            reasoning,
			"sequence_number": s.seqNum,
		}, s.logger)
		s.seqNum++

		// response.reasoning_part.done
		sendSSEEvent(w, map[string]any{
			"type":          "response.reasoning_part.done",
			"item_id":       s.reasoningItemID,
			"output_index":  0,
			"content_index": 0,
			"part": map[string]any{
				"type": "reasoning_text",
				"text": reasoning,
			},
			"sequence_number": s.seqNum,
		}, s.logger)
		s.seqNum++

		// response.output_item.done for reasoning
		sendSSEEvent(w, map[string]any{
			"type":         "response.output_item.done",
			"output_index": 0,
			"item": map[string]any{
				"id":      s.reasoningItemID,
				"type":    "reasoning",
				"summary": []any{},
				"content": []any{
					map[string]any{
						"type": "reasoning_text",
						"text": reasoning,
					},
				},
				"encrypted_content": nil,
				"status":            "completed",
			},
			"sequence_number": s.seqNum,
		}, s.logger)
		s.seqNum++
	}

	// Send message completion events if we started a message
	if s.messageStarted {
		msgOutputIndex := 0
		if s.hasReasoning {
			msgOutputIndex = 1
		}

		// response.output_text.done
		sendSSEEvent(w, map[string]any{
			"type":            "response.output_text.done",
			"item_id":         s.itemID,
			"output_index":    msgOutputIndex,
			"content_index":   0,
			"text":            text,
			"sequence_number": s.seqNum,
		}, s.logger)
		s.seqNum++

		// response.content_part.done
		sendSSEEvent(w, map[string]any{
			"type":          "response.content_part.done",
			"item_id":       s.itemID,
			"output_index":  msgOutputIndex,
			"content_index": 0,
			"part": map[string]any{
				"type":        "output_text",
				"text":        text,
				"annotations": []any{},
				"logprobs":    nil,
			},
			"sequence_number": s.seqNum,
		}, s.logger)
		s.seqNum++

		// response.output_item.done for message
		sendSSEEvent(w, map[string]any{
			"type":         "response.output_item.done",
			"output_index": msgOutputIndex,
			"item": map[string]any{
				"id":   s.itemID,
				"type": "message",
				"role": "assistant",
				"content": []any{
					map[string]any{
						"type":        "output_text",
						"text":        text,
						"annotations": []any{},
						"logprobs":    nil,
					},
				},
				"status": "completed",
				"phase":  nil,
			},
			"sequence_number": s.seqNum,
		}, s.logger)
		s.seqNum++
	}

	// Send tool call completion events in index order
	tcIndices := make([]int, 0, len(s.toolCalls))
	for idx := range s.toolCalls {
		tcIndices = append(tcIndices, idx)
	}
	sort.Ints(tcIndices)
	for _, idx := range tcIndices {
		tc := s.toolCalls[idx]
		if tc.Started {
			args := tc.Arguments.String()

			// response.function_call_arguments.done
			sendSSEEvent(w, map[string]any{
				"type":            "response.function_call_arguments.done",
				"item_id":         tc.ItemID,
				"output_index":    tc.Index,
				"arguments":       args,
				"sequence_number": s.seqNum,
			}, s.logger)
			s.seqNum++

			// response.output_item.done for function_call
			sendSSEEvent(w, map[string]any{
				"type":         "response.output_item.done",
				"output_index": tc.Index,
				"item": map[string]any{
					"id":        tc.ItemID,
					"type":      "function_call",
					"call_id":   tc.ID,
					"name":      tc.Name,
					"arguments": args,
					"status":    "completed",
				},
				"sequence_number": s.seqNum,
			}, s.logger)
			s.seqNum++
		}
	}
}

// convertChatSSEEventToResponses converts a single Chat SSE event to Responses events
func (s *responsesStreamState) convertChatSSEEventToResponses(chatEvent map[string]any) []map[string]any {
	var events []map[string]any

	choices, ok := chatEvent["choices"].([]any)
	if !ok || len(choices) == 0 {
		return events
	}

	choice, ok := choices[0].(map[string]any)
	if !ok {
		return events
	}

	delta, ok := choice["delta"].(map[string]any)
	if !ok {
		return events
	}

	// Handle reasoning content (thinking mode)
	if reasoning, ok := delta["reasoning_content"].(string); ok && reasoning != "" {
		if !s.hasReasoning {
			s.hasReasoning = true
			s.reasoningItemID = fmt.Sprintf("rs_%s", generateSimpleID())

			// Add reasoning item with content array (matching example format)
			events = append(events, map[string]any{
				"type":            "response.output_item.added",
				"output_index":    s.outputIndex,
				"sequence_number": s.seqNum,
				"item": map[string]any{
					"id":                s.reasoningItemID,
					"summary":           []any{},
					"type":              "reasoning",
					"content":           []any{},
					"encrypted_content": nil,
					"status":            nil,
				},
			})
			s.seqNum++
			s.outputIndex++

			// Add reasoning content part
			events = append(events, map[string]any{
				"type":          "response.reasoning_part.added",
				"item_id":       s.reasoningItemID,
				"output_index":  s.outputIndex - 1,
				"content_index": 0,
				"part": map[string]any{
					"type": "reasoning_text",
					"text": "",
				},
				"sequence_number": s.seqNum,
			})
			s.seqNum++
			s.reasoningContentIndex = 0
		}

		// Accumulate reasoning text
		s.reasoningText.WriteString(reasoning)

		// Send reasoning delta
		events = append(events, map[string]any{
			"type":            "response.reasoning_text.delta",
			"item_id":         s.reasoningItemID,
			"output_index":    s.outputIndex - 1,
			"content_index":   0,
			"delta":           reasoning,
			"sequence_number": s.seqNum,
		})
		s.seqNum++
	}

	// Handle content (text)
	if content, ok := delta["content"].(string); ok && content != "" {
		if s.contentIndex == 0 {
			// First content part - add message item
			s.messageStarted = true
			events = append(events, map[string]any{
				"type":            "response.output_item.added",
				"output_index":    s.outputIndex,
				"sequence_number": s.seqNum,
				"item": map[string]any{
					"id":      s.itemID,
					"type":    "message",
					"role":    "assistant",
					"status":  "in_progress",
					"content": []any{},
					"phase":   nil,
				},
			})
			s.seqNum++

			// Add content part
			events = append(events, map[string]any{
				"type":          "response.content_part.added",
				"item_id":       s.itemID,
				"output_index":  s.outputIndex,
				"content_index": s.contentIndex,
				"part": map[string]any{
					"type": "output_text",
				},
				"sequence_number": s.seqNum,
			})
			s.seqNum++
			s.contentIndex++
		}

		// Accumulate text
		s.currentText.WriteString(content)

		// Send text delta
		events = append(events, map[string]any{
			"type":            "response.output_text.delta",
			"item_id":         s.itemID,
			"output_index":    s.outputIndex,
			"content_index":   0,
			"delta":           content,
			"sequence_number": s.seqNum,
		})
		s.seqNum++
	}

	// Handle tool calls
	if tcArray, ok := delta["tool_calls"].([]any); ok {
		for _, tc := range tcArray {
			tcMap, ok := tc.(map[string]any)
			if !ok {
				continue
			}

			// Get tool call index (Chat Completions uses index for parallel tool calls)
			tcIndex := 0
			if idx, ok := tcMap["index"].(float64); ok {
				tcIndex = int(idx)
			}

			// Get or create tool call state
			tcState, exists := s.toolCalls[tcIndex]
			if !exists {
				tcState = &toolCallState{
					Index: s.outputIndex,
				}
				s.toolCalls[tcIndex] = tcState
			}

			// Extract tool call ID (only present in first chunk)
			if id, ok := tcMap["id"].(string); ok && id != "" {
				tcState.ID = id
			}

			// Extract function details
			if fn, ok := tcMap["function"].(map[string]any); ok {
				// Function name (only in first chunk)
				if name, ok := fn["name"].(string); ok && name != "" {
					tcState.Name = name
				}

				// Arguments delta (streamed)
				if args, ok := fn["arguments"].(string); ok && args != "" {
					tcState.Arguments.WriteString(args)

					// If this is the first chunk with data, emit output_item.added
					if !tcState.Started {
						tcState.Started = true
						tcState.ItemID = fmt.Sprintf("fc_%s", generateSimpleID())
						tcState.Index = s.outputIndex
						s.outputIndex++

						events = append(events, map[string]any{
							"type":            "response.output_item.added",
							"output_index":    tcState.Index,
							"sequence_number": s.seqNum,
							"item": map[string]any{
								"id":        tcState.ItemID,
								"type":      "function_call",
								"call_id":   tcState.ID,
								"name":      tcState.Name,
								"arguments": "",
								"status":    "in_progress",
							},
						})
						s.seqNum++
					}

					// Emit arguments delta
					events = append(events, map[string]any{
						"type":            "response.function_call_arguments.delta",
						"item_id":         tcState.ItemID,
						"output_index":    tcState.Index,
						"delta":           args,
						"sequence_number": s.seqNum,
					})
					s.seqNum++
				}
			}
		}
	}

	return events
}

// buildInitialResponse builds the initial response object for streaming
func buildInitialResponse(responseID, model string, createdAt int64, status string) map[string]any {
	return map[string]any{
		"id":                   responseID,
		"created_at":           createdAt,
		"incomplete_details":   nil,
		"instructions":         nil,
		"metadata":             nil,
		"model":                model,
		"object":               "response",
		"output":               []any{},
		"parallel_tool_calls":  true,
		"temperature":          nil,
		"tool_choice":          "auto",
		"tools":                []any{},
		"top_p":                nil,
		"background":           false,
		"max_output_tokens":    nil,
		"max_tool_calls":       nil,
		"previous_response_id": nil,
		"prompt":               nil,
		"reasoning":            nil,
		"service_tier":         "auto",
		"status":               status,
		"text":                 nil,
		"top_logprobs":         nil,
		"truncation":           "disabled",
		"usage": map[string]any{
			"input_tokens":  0,
			"output_tokens": 0,
			"total_tokens":  0,
		},
		"user":            nil,
		"input_messages":  nil,
		"output_messages": nil,
	}
}

// buildFinalResponse builds the final response object for streaming completion
func buildFinalResponse(responseID, itemID, reasoningItemID, model string, createdAt int64, text, reasoning string, usage map[string]any, finishReason string, toolCalls map[int]*toolCallState) map[string]any {
	output := []any{}

	// Add reasoning item first if present (matching example format)
	if reasoning != "" {
		rsID := reasoningItemID
		if rsID == "" {
			rsID = fmt.Sprintf("rs_%s", generateSimpleID())
		}
		reasoningItem := map[string]any{
			"id":      rsID,
			"summary": []any{},
			"type":    "reasoning",
			"content": []any{
				map[string]any{
					"text": reasoning,
					"type": "reasoning_text",
				},
			},
			"encrypted_content": nil,
			"status":            nil,
		}
		output = append(output, reasoningItem)
	}

	// Build message item with content (only if there's text content)
	if text != "" {
		contentParts := []any{
			map[string]any{
				"annotations": []any{},
				"text":        text,
				"type":        "output_text",
				"logprobs":    nil,
			},
		}
		messageItem := map[string]any{
			"id":      itemID,
			"content": contentParts,
			"role":    "assistant",
			"status":  "completed",
			"type":    "message",
			"phase":   nil,
		}
		output = append(output, messageItem)
	}

	// Add tool calls to output in index order
	tcIndices := make([]int, 0, len(toolCalls))
	for idx := range toolCalls {
		tcIndices = append(tcIndices, idx)
	}
	sort.Ints(tcIndices)
	for _, idx := range tcIndices {
		tc := toolCalls[idx]
		if tc.Started {
			toolCallItem := map[string]any{
				"id":        tc.ItemID,
				"type":      "function_call",
				"call_id":   tc.ID,
				"name":      tc.Name,
				"arguments": tc.Arguments.String(),
				"status":    "completed",
			}
			output = append(output, toolCallItem)
		}
	}

	// Determine status based on finish_reason
	status := "completed"
	var incompleteDetails map[string]any
	if finishReason == "length" {
		status = "incomplete"
		incompleteDetails = map[string]any{
			"reason": "max_output_tokens",
		}
	}

	return map[string]any{
		"id":                   responseID,
		"created_at":           createdAt,
		"incomplete_details":   incompleteDetails,
		"instructions":         nil,
		"metadata":             nil,
		"model":                model,
		"object":               "response",
		"output":               output,
		"parallel_tool_calls":  true,
		"temperature":          nil,
		"tool_choice":          "auto",
		"tools":                []any{},
		"top_p":                nil,
		"background":           false,
		"max_output_tokens":    nil,
		"max_tool_calls":       nil,
		"previous_response_id": nil,
		"prompt":               nil,
		"reasoning":            nil,
		"service_tier":         "auto",
		"status":               status,
		"text":                 nil,
		"top_logprobs":         nil,
		"truncation":           "disabled",
		"usage":                convertUsage(usage),
		"user":                 nil,
		"input_messages":       nil,
		"output_messages":      nil,
	}
}

// sendSSEEvent writes an SSE event to the response
func sendSSEEvent(w http.ResponseWriter, event map[string]any, logger *slog.Logger) {
	jsonBytes, err := json.Marshal(event)
	if err != nil {
		logger.Error("failed to marshal SSE event", slog.Any("error", err))
		return
	}

	if _, err := fmt.Fprintf(w, "data: %s\n\n", jsonBytes); err != nil {
		logger.Error("failed to write SSE event", slog.Any("error", err))
		return
	}

	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
}

// idCounter is an atomic counter to ensure unique IDs even when called in tight succession
var idCounter atomic.Uint64

// generateSimpleID generates a unique ID using timestamp + atomic counter
func generateSimpleID() string {
	count := idCounter.Add(1)
	return fmt.Sprintf("%d_%d", time.Now().UnixNano()%1000000000000, count)
}

