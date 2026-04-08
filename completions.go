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

// legacyCompletions handles /v1/completions (text completions API).
// Unlike chat completions, this endpoint uses raw prompts with no chat template.
// We only validate the virtual model name, swap it to the served model, and fix
// the model name in the response. No sampling params or chat_template_kwargs.
func legacyCompletions(httpCli *http.Client, target *url.URL,
	servedModel, thinkingGeneral, thinkingCoding, instructGeneral, instructReasoning string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		logger := logger.With(httplog.GetReqIDSLogAttr(r.Context()))
		ctx := r.Context()
		// Read request body
		r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodySize)
		requestBody, err := io.ReadAll(r.Body)
		if err != nil {
			logger.Error("failed to read body", slog.String("error", err.Error()))
			httpError(ctx, w, readBodyStatusCode(err))
			return
		}
		// Parse request body
		var data map[string]any
		if err = json.Unmarshal(requestBody, &data); err != nil {
			logger.Error("failed to parse body as JSON", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusBadRequest)
			return
		}
		modelName, ok := data["model"].(string)
		if !ok {
			logger.Error("missing/invalid model in request body")
			httpError(ctx, w, http.StatusBadRequest)
			return
		}
		// Validate virtual model name
		switch modelName {
		case thinkingGeneral, thinkingCoding, instructGeneral, instructReasoning:
			logger.Info("legacy completions model matched", slog.String("virtual_model", modelName))
		default:
			logger.Error("unsupported model", slog.String("model", modelName))
			httpError(ctx, w, http.StatusBadRequest)
			return
		}
		// Track streaming mode for response fixing
		var stream bool
		if streamVal, ok := data["stream"]; ok {
			stream, _ = streamVal.(bool)
		}
		// Swap model name, preserve everything else
		virtualModel := modelName
		data["model"] = servedModel
		requestBody, err = json.Marshal(data)
		if err != nil {
			logger.Error("failed to marshal request body", slog.Any("error", err))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}
		logger.Debug("rewritten request body", slog.String("body", string(requestBody)))
		modifiedRequests.Add(1)
		// Prepare and send outgoing request
		outreq := r.Clone(ctx)
		rewriteRequestURL(outreq, target)
		stripHopByHopHeaders(outreq)
		outreq.Body = io.NopCloser(bytes.NewReader(requestBody))
		outreq.ContentLength = int64(len(requestBody))
		outreq.RequestURI = ""
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
		if stream && outResp.StatusCode >= 200 && outResp.StatusCode < 300 {
			logger.Debug("streaming response to client with model name fix")
			copyHeaders(w, outResp)
			w.WriteHeader(outResp.StatusCode)
			if err = streamResponse(w, outResp.Body, virtualModel, logger); err != nil {
				logger.Error("failed to stream response", slog.String("error", err.Error()))
			}
		} else if stream {
			logger.Warn("backend returned error for streaming request, passing through raw response",
				slog.Int("status", outResp.StatusCode),
			)
			copyHeaders(w, outResp)
			w.WriteHeader(outResp.StatusCode)
			if _, err = io.Copy(w, outResp.Body); err != nil {
				logger.Error("failed to write error response", slog.String("error", err.Error()))
			}
		} else {
			responseBody, err := io.ReadAll(outResp.Body)
			if err != nil {
				logger.Error("failed to read response body", slog.String("error", err.Error()))
				httpError(ctx, w, http.StatusInternalServerError)
				return
			}
			// Only fix model name on success; pass through errors as-is
			if outResp.StatusCode >= 200 && outResp.StatusCode < 300 {
				responseBody = fixModelNameInResponse(responseBody, virtualModel, logger)
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

// fixModelNameInResponse replaces the backend model name with the virtual model name in a JSON response.
func fixModelNameInResponse(responseBody []byte, virtualModel string, logger *slog.Logger) []byte {
	var data map[string]any
	if err := json.Unmarshal(responseBody, &data); err != nil {
		return responseBody
	}
	if modelStr, ok := data["model"].(string); ok {
		logger.Debug("fixing model name in response",
			slog.String("original", modelStr),
			slog.String("replacement", virtualModel),
		)
		data["model"] = virtualModel
		fixedBody, err := json.Marshal(data)
		if err != nil {
			return responseBody
		}
		return fixedBody
	}
	return responseBody
}

func transform(httpCli *http.Client, target *url.URL,
	servedModel, thinkingGeneral, thinkingCoding, instructGeneral, instructReasoning string, enforceSamplingParams bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Prepare
		logger := logger.With(httplog.GetReqIDSLogAttr(r.Context()))
		ctx := r.Context()
		var think, stream bool // Track thinking mode and streaming for response fixing
		// Read request body
		r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodySize)
		requestBody, err := io.ReadAll(r.Body)
		if err != nil {
			logger.Error("failed to read body", slog.String("error", err.Error()))
			httpError(ctx, w, readBodyStatusCode(err))
			return
		}
		// Parse request body
		var data map[string]any
		err = json.Unmarshal(requestBody, &data)
		if err != nil {
			logger.Error("failed to parse body as JSON", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusBadRequest)
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
		if ok && kwargs != nil {
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
		logger.Debug("rewritten request body", slog.String("body", string(requestBody)))
		// Track modified request
		modifiedRequests.Add(1)
		// prepare outgoing request
		outreq := r.Clone(ctx)
		rewriteRequestURL(outreq, target)
		stripHopByHopHeaders(outreq)
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

		if stream && outResp.StatusCode >= 200 && outResp.StatusCode < 300 {
			// Streaming mode: copy headers and proxy response body with model name fixing
			logger.Debug("streaming response to client with model name fix")
			copyHeaders(w, outResp)
			w.WriteHeader(outResp.StatusCode)
			if err = streamResponse(w, outResp.Body, virtualModel, logger); err != nil {
				logger.Error("failed to stream response", slog.String("error", err.Error()))
			}
		} else if stream {
			// Backend returned an error for a streaming request: pass through the raw error body
			logger.Warn("backend returned error for streaming request, passing through raw response",
				slog.Int("status", outResp.StatusCode),
			)
			copyHeaders(w, outResp)
			w.WriteHeader(outResp.StatusCode)
			if _, err = io.Copy(w, outResp.Body); err != nil {
				logger.Error("failed to write error response", slog.String("error", err.Error()))
			}
		} else {
			// Non-streaming mode: read full response, fix bugs, then write
			responseBody, err := io.ReadAll(outResp.Body)
			if err != nil {
				logger.Error("failed to read response body", slog.String("error", err.Error()))
				httpError(ctx, w, http.StatusInternalServerError)
				return
			}

			// Only attempt JSON fixes on success responses; pass through errors as-is
			if outResp.StatusCode >= 200 && outResp.StatusCode < 300 {
				responseBody = fixNonStreamingResponse(responseBody, think, virtualModel, logger)
			} else {
				logger.Warn("backend returned error for non-streaming request, passing through raw response",
					slog.Int("status", outResp.StatusCode),
				)
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

// fixNonStreamingResponse fixes the non-streaming response in a single JSON pass:
//   - Replaces the backend model name with the virtual model name
//   - When think=false, moves misplaced reasoning_content/reasoning to content (vLLM bug)
func fixNonStreamingResponse(responseBody []byte, think bool, virtualModel string, logger *slog.Logger) []byte {
	var data map[string]any
	if err := json.Unmarshal(responseBody, &data); err != nil {
		return responseBody
	}

	modified := false

	// Fix model name
	if modelStr, ok := data["model"].(string); ok {
		logger.Debug("fixing model name in response",
			slog.String("original", modelStr),
			slog.String("replacement", virtualModel),
		)
		data["model"] = virtualModel
		modified = true
	}

	// Fix vLLM bug: non-thinking responses incorrectly placed in reasoning_content/reasoning
	if !think {
		if fixReasoningContentBug(data, logger) {
			modified = true
		}
	}

	if !modified {
		return responseBody
	}

	fixedBody, err := json.Marshal(data)
	if err != nil {
		logger.Error("failed to marshal fixed response body", slog.Any("error", err))
		return responseBody
	}
	return fixedBody
}

// fixReasoningContentBug fixes a vLLM bug where non-thinking responses have content
// incorrectly placed in reasoning_content/reasoning instead of content.
// Operates in-place on a parsed Chat Completions response map. Returns true if any fix was applied.
func fixReasoningContentBug(data map[string]any, logger *slog.Logger) bool {
	choices, ok := data["choices"].([]any)
	if !ok {
		return false
	}

	fixed := false
	for i, choice := range choices {
		choiceMap, ok := choice.(map[string]any)
		if !ok {
			continue
		}

		message, ok := choiceMap["message"].(map[string]any)
		if !ok {
			continue
		}

		// Check if content is empty/missing and reasoning_content or reasoning exists
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
				delete(message, "reasoning_content")
				delete(message, "reasoning")
				choiceMap["message"] = message
				choices[i] = choiceMap
				fixed = true
				logger.Info("vLLM response fixed: moved reasoning content to content field",
					slog.String("source_field", reasoningSource),
					slog.Int("choice_index", i),
				)
			}
		}
	}
	return fixed
}

// streamResponse streams SSE events from backend to client, fixing model name in all events.
// Note: no explicit Flush() call is needed here — the httplog middleware wraps the ResponseWriter
// and auto-flushes on every Write() when Content-Type is a streamable type (e.g. text/event-stream).
func streamResponse(w http.ResponseWriter, backendBody io.ReadCloser, virtualModel string, logger *slog.Logger) error {
	buf := make([]byte, 0, 4096)
	temp := make([]byte, 4096)

	for {
		n, err := backendBody.Read(temp)
		if n > 0 {
			buf = append(buf, temp[:n]...)
		}
		if err != nil && err != io.EOF {
			return err
		}

		// Guard against unbounded buffer growth from malformed streams
		if len(buf) > maxSSEEventSize {
			return fmt.Errorf("SSE buffer exceeded maximum size (%d bytes)", maxSSEEventSize)
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
			event = fixModelNameInSSE(event, virtualModel, logger)

			if _, werr := w.Write(event); werr != nil {
				return werr
			}
		}

		if err == io.EOF {
			// Attempt to fix model name in any remaining data before writing
			if len(buf) > 0 {
				buf = fixModelNameInSSE(buf, virtualModel, logger)
				if _, werr := w.Write(buf); werr != nil {
					return werr
				}
			}
			return nil
		}
	}
}

// fixModelNameInSSE fixes the model field in an SSE event while preserving
// all SSE fields (id:, event:, retry:, etc.). Only the data: line containing
// JSON with a model field is modified.
func fixModelNameInSSE(event []byte, virtualModel string, logger *slog.Logger) []byte {
	// Split event into lines, preserving structure
	// Event includes trailing \n\n — trim it for processing, re-add at end
	trimmed := bytes.TrimRight(event, "\n")
	lines := bytes.Split(trimmed, []byte("\n"))

	modified := false
	for i, line := range lines {
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}

		jsonPart := bytes.TrimSpace(line[6:])

		// Skip [DONE] or empty data lines
		if len(jsonPart) == 0 || bytes.Equal(jsonPart, []byte("[DONE]")) {
			continue
		}

		// Try to parse and fix
		var data map[string]any
		if err := json.Unmarshal(jsonPart, &data); err != nil {
			continue
		}

		if modelStr, ok := data["model"].(string); ok {
			logger.Debug("fixing model name in streaming event",
				slog.String("original", modelStr),
				slog.String("replacement", virtualModel),
			)
			data["model"] = virtualModel

			fixedJSON, err := json.Marshal(data)
			if err != nil {
				logger.Error("failed to marshal streaming event", slog.Any("error", err))
				continue
			}

			lines[i] = append([]byte("data: "), fixedJSON...)
			modified = true
		}
	}

	if !modified {
		return event
	}

	// Reassemble event with original \n\n terminator
	result := bytes.Join(lines, []byte("\n"))
	result = append(result, []byte("\n\n")...)
	return result
}
