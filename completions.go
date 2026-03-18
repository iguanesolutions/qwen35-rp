package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"syscall"

	"github.com/hekmon/httplog/v3"
)

var (
	// Thinking mode for general tasks
	thinkingGeneralParams = map[string]any{
		"temperature":        1.0,
		"top_p":              0.95,
		"top_k":              20,
		"min_p":              0.0,
		"presence_penalty":   1.5,
		"repetition_penalty": 1.0,
	}
	// Thinking mode for precise coding tasks
	thinkingCodingParams = map[string]any{
		"temperature":        0.6,
		"top_p":              0.95,
		"top_k":              20,
		"min_p":              0.0,
		"presence_penalty":   0.0,
		"repetition_penalty": 1.0,
	}
	// Instruct mode for general tasks
	instructGeneralParams = map[string]any{
		"temperature":        0.7,
		"top_p":              0.8,
		"top_k":              20,
		"min_p":              0.0,
		"presence_penalty":   1.5,
		"repetition_penalty": 1.0,
	}
	// Instruct mode for reasoning tasks
	instructReasoningParams = map[string]any{
		"temperature":        1.0,
		"top_p":              0.95,
		"top_k":              20,
		"min_p":              0.0,
		"presence_penalty":   1.5,
		"repetition_penalty": 1.0,
	}
)

func transform(httpCli *http.Client, target *url.URL,
	servedModel, thinkingGeneral, thinkingCoding, instructGeneral, instructReasoning string, enforceSamplingParams bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Prepare
		logger := logger.With(httplog.GetReqIDSLogAttr(r.Context()))
		ctx := r.Context()
		var think, stream bool // Track thinking mode and streaming for response fixing
		// Read request body
		requestBody, err := io.ReadAll(r.Body)
		if err != nil {
			logger.Error("failed to read body", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}
		// Parse request body
		var data map[string]any
		err = json.Unmarshal(requestBody, &data)
		if err != nil {
			logger.Error("failed to parse body as JSON", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}
		modelName, ok := data["model"].(string)
		if !ok {
			logger.Error("missing/invalid model in request body")
			httpError(ctx, w, http.StatusBadRequest)
			return
		}
		// Track streaming mode for response fixing
		if streamVal, ok := data["stream"]; ok {
			stream, _ = streamVal.(bool)
		}
		// check thinking mode based on model name and apply sampling parameters
		switch modelName {
		case thinkingGeneral:
			think = true
			logger.Info("model matched",
				slog.String("type", "thinking_general"),
				slog.String("virtual_model", modelName),
			)
			applySamplingParams(data, thinkingGeneralParams, logger, enforceSamplingParams)
		case thinkingCoding:
			think = true
			logger.Info("model matched",
				slog.String("type", "thinking_coding"),
				slog.String("virtual_model", modelName),
			)
			applySamplingParams(data, thinkingCodingParams, logger, enforceSamplingParams)
		case instructGeneral:
			think = false
			logger.Info("model matched",
				slog.String("type", "instruct_general"),
				slog.String("virtual_model", modelName),
			)
			applySamplingParams(data, instructGeneralParams, logger, enforceSamplingParams)
		case instructReasoning:
			think = false
			logger.Info("model matched",
				slog.String("type", "instruct_reasoning"),
				slog.String("virtual_model", modelName),
			)
			applySamplingParams(data, instructReasoningParams, logger, enforceSamplingParams)
		default:
			logger.Error("unsupported model", slog.String("model", modelName))
			httpError(ctx, w, http.StatusBadRequest)
			return
		}
		// Track the virtual model name requested by client (before override)
		virtualModel := modelName
		// override model name for backend
		data["model"] = servedModel
		// set thinking extra body parameter
		kwargs, ok := data["chat_template_kwargs"]
		if ok {
			kwargsMap, ok := kwargs.(map[string]any)
			if !ok {
				logger.Error("chat_template_kwargs is not a map[string]any")
				httpError(ctx, w, http.StatusBadRequest)
				return
			}
			kwargsMap["enable_thinking"] = think
			data["chat_template_kwargs"] = kwargsMap
		} else {
			data["chat_template_kwargs"] = map[string]any{"enable_thinking": think}
		}
		// marshal request body
		requestBody, err = json.Marshal(data)
		if err != nil {
			logger.Error("failed to marshal request body", slog.Any("error", err))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}
		logger.Debug("rewrited request body", slog.String("body", string(requestBody)))
		// Track modified request
		modifiedRequests.Add(1)
		// prepare outgoing request
		outreq := r.Clone(ctx)
		rewriteRequestURL(outreq, target)
		outreq.Body = io.NopCloser(bytes.NewReader(requestBody))
		outreq.ContentLength = int64(len(requestBody))
		outreq.RequestURI = ""
		// send request
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

		for header, values := range outResp.Header {
			for _, value := range values {
				w.Header().Add(header, value)
			}
		}

		if stream {
			// Streaming mode: proxy response body with model name fixing
			logger.Debug("streaming response to client with model name fix")
			w.WriteHeader(outResp.StatusCode)
			if err = streamResponse(w, outResp.Body, virtualModel, logger); err != nil {
				logger.Error("failed to stream response", slog.String("error", err.Error()))
			}
		} else {
			// Non-streaming mode: read full response, fix bugs, then write
			responseBody, err := io.ReadAll(outResp.Body)
			if err != nil {
				logger.Error("failed to read response body", slog.String("error", err.Error()))
				httpError(ctx, w, http.StatusInternalServerError)
				return
			}

			// Fix vLLM bug: non-thinking, non-streaming responses incorrectly placed in reasoning_content/reasoning fields
			responseBody = fixVLLMResponse(responseBody, think, stream, logger)

			// Fix model name in response: replace backend model with virtual model name
			responseBody = fixModelName(responseBody, virtualModel, logger)

			w.Header().Set("Content-Length", strconv.Itoa(len(responseBody)))
			w.WriteHeader(outResp.StatusCode)
			if _, err = w.Write(responseBody); err != nil {
				logger.Error("failed to write response", slog.String("error", err.Error()))
			}
		}
	}
}

// fixVLLMResponse fixes vLLM bug where non-thinking responses are incorrectly
// placed in reasoning_content or reasoning fields instead of content field.
// This only applies when think=false (no-thinking mode) AND stream=false (non-streaming).
func fixVLLMResponse(responseBody []byte, think, stream bool, logger *slog.Logger) []byte {
	// Only fix responses for non-thinking mode and non-streaming
	if think || stream {
		return responseBody
	}

	// Try to parse the response as JSON
	var data map[string]any
	if err := json.Unmarshal(responseBody, &data); err != nil {
		// Not valid JSON, return as-is
		return responseBody
	}

	// Check if this is a chat completion response (has choices array)
	choices, ok := data["choices"].([]any)
	if !ok || len(choices) == 0 {
		return responseBody
	}

	modified := false
	for i, choice := range choices {
		choiceMap, ok := choice.(map[string]any)
		if !ok {
			continue
		}

		// Check if this is a delta (streaming) or message (non-streaming)
		var message map[string]any
		var isDelta bool

		if msg, ok := choiceMap["message"].(map[string]any); ok {
			message = msg
			isDelta = false
		} else if delta, ok := choiceMap["delta"].(map[string]any); ok {
			message = delta
			isDelta = true
		} else {
			continue
		}

		// Check if content is empty/missing and reasoning_content or reasoning exists
		// Handle content as string or null
		var content string
		var hasContent bool
		if contentVal, exists := message["content"]; exists {
			if contentStr, ok := contentVal.(string); ok {
				content = contentStr
				hasContent = true
			}
		}
		reasoningContent, hasReasoningContent := message["reasoning_content"].(string)
		reasoning, hasReasoning := message["reasoning"].(string)

		// Fix: if content is empty/missing but reasoning_content or reasoning has value, move it to content
		if (!hasContent || content == "") && (hasReasoningContent || hasReasoning) {
			var reasoningText string
			var reasoningSource string
			if hasReasoningContent && reasoningContent != "" {
				reasoningText = reasoningContent
				reasoningSource = "reasoning_content"
			} else if hasReasoning && reasoning != "" {
				reasoningText = reasoning
				reasoningSource = "reasoning"
			}

			if reasoningText != "" {
				message["content"] = reasoningText
				// Remove the incorrect fields
				delete(message, "reasoning_content")
				delete(message, "reasoning")
				modified = true
				logger.Info("vLLM response fixed: moved reasoning content to content field (no-thinking, non-streaming mode)",
					slog.String("source_field", reasoningSource),
					slog.Int("choice_index", i),
				)
			}
		}

		// Update the choice in the array
		if isDelta {
			choiceMap["delta"] = message
		} else {
			choiceMap["message"] = message
		}
		choices[i] = choiceMap
	}

	if modified {
		data["choices"] = choices
		// Re-marshal the response
		fixedBody, err := json.Marshal(data)
		if err != nil {
			logger.Error("failed to marshal fixed response body", slog.Any("error", err))
			return responseBody
		}
		logger.Debug("response body re-marshaled after fix")
		return fixedBody
	}

	return responseBody
}

// fixModelName replaces the backend model name with the virtual model name
// that the client originally requested
func fixModelName(responseBody []byte, virtualModel string, logger *slog.Logger) []byte {
	// Try to parse the response as JSON
	var data map[string]any
	if err := json.Unmarshal(responseBody, &data); err != nil {
		// Not valid JSON, return as-is
		return responseBody
	}

	// Check if model field exists and replace it
	if _, ok := data["model"]; ok {
		logger.Debug("fixing model name in response",
			slog.String("original", data["model"].(string)),
			slog.String("replacement", virtualModel),
		)
		data["model"] = virtualModel

		// Re-marshal the response
		fixedBody, err := json.Marshal(data)
		if err != nil {
			logger.Error("failed to marshal response with fixed model name", slog.Any("error", err))
			return responseBody
		}
		return fixedBody
	}

	return responseBody
}

// streamResponse streams SSE events from backend to client, fixing model name in all events
func streamResponse(w http.ResponseWriter, backendBody io.ReadCloser, virtualModel string, logger *slog.Logger) error {
	buf := make([]byte, 0, 4096)
	temp := make([]byte, 4096)

	for {
		n, err := backendBody.Read(temp)
		if n > 0 {
			buf = append(buf, temp[:n]...)
		}
		if err != nil {
			if err == io.EOF {
				// Write any remaining data
				if len(buf) > 0 {
					if _, werr := w.Write(buf); werr != nil {
						return werr
					}
				}
				return nil
			}
			return err
		}

		// Process complete SSE events (separated by double newline)
		for {
			idx := bytes.Index(buf, []byte("\n\n"))
			if idx == -1 {
				break // Need more data
			}

			event := buf[:idx+2] // Include the \n\n
			buf = buf[idx+2:]

			// Fix model name in ALL data events (backend includes model in every chunk)
			if bytes.HasPrefix(event, []byte("data: ")) {
				event = fixModelNameInSSE(event, virtualModel, logger)
			}

			if _, werr := w.Write(event); werr != nil {
				return werr
			}
		}

		// Prevent buffer from growing too large
		if len(buf) > 8192 {
			if _, werr := w.Write(buf); werr != nil {
				return werr
			}
			buf = buf[:0]
		}
	}
}

// fixModelNameInSSE fixes the model field in an SSE data event
func fixModelNameInSSE(event []byte, virtualModel string, logger *slog.Logger) []byte {
	// Extract JSON part (after "data: ")
	if !bytes.HasPrefix(event, []byte("data: ")) {
		return event
	}

	jsonPart := event[6:] // Skip "data: "
	jsonPart = bytes.TrimSpace(jsonPart)

	// Skip [DONE] or empty events
	if len(jsonPart) == 0 || bytes.Equal(jsonPart, []byte("[DONE]")) {
		return event
	}

	// Try to parse and fix
	var data map[string]any
	if err := json.Unmarshal(jsonPart, &data); err != nil {
		// Not valid JSON, return original
		return event
	}

	// Fix model field if present
	if _, ok := data["model"]; ok {
		if modelStr, ok := data["model"].(string); ok {
			logger.Debug("fixing model name in streaming event",
				slog.String("original", modelStr),
				slog.String("replacement", virtualModel),
			)
			data["model"] = virtualModel

			// Re-marshal
			fixedJSON, err := json.Marshal(data)
			if err != nil {
				logger.Error("failed to marshal streaming event", slog.Any("error", err))
				return event
			}

			// Reconstruct SSE event
			return append([]byte("data: "), append(fixedJSON, []byte("\n\n")...)...)
		}
	}

	return event
}
