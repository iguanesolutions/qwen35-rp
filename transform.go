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
		// Track thinking mode and streaming for response fixing
		var think, stream bool
		// Prepare
		logger := logger.With(httplog.GetReqIDSLogAttr(r.Context()))
		logger.Info("received a request",
			slog.String("remote_addr", r.RemoteAddr),
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
		)
		ctx := r.Context()
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
			applySamplingParams(data, thinkingGeneralParams, logger, enforceSamplingParams)
		case thinkingCoding:
			think = true
			applySamplingParams(data, thinkingCodingParams, logger, enforceSamplingParams)
		case instructGeneral:
			think = false
			applySamplingParams(data, instructGeneralParams, logger, enforceSamplingParams)
		case instructReasoning:
			think = false
			applySamplingParams(data, instructReasoningParams, logger, enforceSamplingParams)
		default:
			logger.Error("unsupported model", slog.String("model", modelName))
			httpError(ctx, w, http.StatusBadRequest)
			return
		}
		// override model name
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

		// Read response body
		responseBody, err := io.ReadAll(outResp.Body)
		if err != nil {
			logger.Error("failed to read response body", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}

		// Fix vLLM bug: non-thinking, non-streaming responses incorrectly placed in reasoning_content/reasoning fields
		responseBody = fixVLLMResponse(responseBody, think, stream, logger)

		for header, values := range outResp.Header {
			for _, value := range values {
				w.Header().Add(header, value)
			}
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(responseBody)))
		w.WriteHeader(outResp.StatusCode)
		if _, err = w.Write(responseBody); err != nil {
			logger.Error("failed to write response", slog.String("error", err.Error()))
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
		content, hasContent := message["content"].(string)
		reasoningContent, hasReasoningContent := message["reasoning_content"].(string)
		reasoning, hasReasoning := message["reasoning"].(string)

		// Fix: if content is empty but reasoning_content or reasoning has value, move it to content
		if (!hasContent || content == "") && (hasReasoningContent || hasReasoning) {
			var reasoningText string
			if hasReasoningContent {
				reasoningText = reasoningContent
			} else if hasReasoning {
				reasoningText = reasoning
			}

			if reasoningText != "" {
				message["content"] = reasoningText
				// Remove the incorrect fields
				delete(message, "reasoning_content")
				delete(message, "reasoning")
				modified = true
				logger.Info("vLLM response fixed: moved reasoning content to content field (no-thinking, non-streaming mode)")
				logger.Debug("fixed vLLM response: moved reasoning content to content field")
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
		return fixedBody
	}

	return responseBody
}
