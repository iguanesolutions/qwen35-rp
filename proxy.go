package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"path"
	"strings"
	"syscall"

	"github.com/hekmon/httplog/v2"
)

const (
	chatCompletionsURI        = "/v1/chat/completions"          // Path to intercept for chat completions
	noThinkChatCompletionsURI = "/nothink" + chatCompletionsURI // Path to intercept for forced nothink chat completions
	thinkChatCompletionsURI   = "/think" + chatCompletionsURI   // Path to intercept for forced think chat completions

	contentTypeHeader       = "Content-Type"     // Header key for content type
	MIMETypeApplicationJSON = "application/json" // Value for JSON content type
)

type mode int

const (
	modeAuto mode = iota
	modeNoThink
	modeThink
)

func (m mode) String() string {
	switch m {
	case modeAuto:
		return "auto"
	case modeNoThink:
		return "no_think"
	case modeThink:
		return "think"
	default:
		return fmt.Sprintf("unknown(%d)", m)
	}
}

const (
	maxTokensKey       = "max_tokens"
	maxTokens          = 8192
	temperatureKey     = "temperature"
	thinkTemperature   = 0.6
	noThinkTemperature = 0.7
	topPKey            = "top_p"
	thinkTopP          = 0.95
	noThinkTopP        = 0.8
	topKKey            = "top_k"
	TopK               = 20
	minPKey            = "min_p"
	minP               = 0.0
	presencePenaltyKey = "presence_penalty"
	presencePenalty    = 1.5

	thinkSwitch   = "/think"
	noThinkSwitch = "/nothink"
)

var (
	suffixes      = []string{thinkSwitch, noThinkSwitch}
	suffixLengths = []int{len(thinkSwitch), len(noThinkSwitch)}
)

func proxy(target *url.URL) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		logger := logger.With(httplog.GetReqIDSLogAttr(r.Context()))
		logger.Info("received a request",
			slog.String("remote_addr", r.RemoteAddr),
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
		)
		var err error
		// Inspect and modify body if necessary
		var targetPath string
		switch r.URL.Path {
		case chatCompletionsURI:
			// detect mode
			if strings.HasPrefix(r.Header.Get(contentTypeHeader), MIMETypeApplicationJSON) {
				// Replace request body (that will be proxified) by the inspected one (that might be modified)
				var detectedMode mode
				if r.Body, detectedMode, err = deepRequestInspection(r.Body, modeAuto, logger); err != nil {
					logger.Error("failed to inspect request body", slog.Any("error", err))
					http.Error(w,
						generateErrorClientText(r.Context(), http.StatusInternalServerError),
						http.StatusInternalServerError,
					)
					return
				} else {
					logger.Info("detected mode", slog.String("mode", detectedMode.String()))
				}
			} else {
				logger.Warn("unsupported content type for automatic chat completions",
					slog.String("content_type", r.Header.Get(contentTypeHeader)),
					slog.String("expected_prefix", MIMETypeApplicationJSON),
				)
			}
			targetPath = chatCompletionsURI
		case noThinkChatCompletionsURI:
			// force nothink
			if strings.HasPrefix(r.Header.Get(contentTypeHeader), MIMETypeApplicationJSON) {
				// Replace request body (that will be proxified) by the inspected one (that might be modified)
				if r.Body, _, err = deepRequestInspection(r.Body, modeNoThink, logger); err != nil {
					logger.Error("failed to inspect request body", slog.Any("error", err))
					http.Error(w,
						generateErrorClientText(r.Context(), http.StatusInternalServerError),
						http.StatusInternalServerError,
					)
					return
				} else {
					logger.Info("forcing mode", slog.String("mode", modeNoThink.String()))
				}
			} else {
				logger.Warn("unsupported content type for force no think chat completions",
					slog.String("content_type", r.Header.Get(contentTypeHeader)),
					slog.String("expected_prefix", MIMETypeApplicationJSON),
				)
			}
			targetPath = chatCompletionsURI
		case thinkChatCompletionsURI:
			// force think
			if strings.HasPrefix(r.Header.Get(contentTypeHeader), MIMETypeApplicationJSON) {
				// Replace request body (that will be proxified) by the inspected one (that might be modified)
				if r.Body, _, err = deepRequestInspection(r.Body, modeThink, logger); err != nil {
					logger.Error("failed to inspect request body", slog.Any("error", err))
					http.Error(w,
						generateErrorClientText(r.Context(), http.StatusInternalServerError),
						http.StatusInternalServerError,
					)
					return
				} else {
					logger.Info("forcing mode", slog.String("mode", modeThink.String()))
				}
			} else {
				logger.Warn("unsupported content type for force think chat completions",
					slog.String("content_type", r.Header.Get(contentTypeHeader)),
					slog.String("expected_prefix", MIMETypeApplicationJSON),
				)
			}
			targetPath = chatCompletionsURI
		default:
			targetPath = r.URL.Path
			logger.Debug("proxying request without modification")
		}
		// Create the upstream request
		upstreamURL := *target
		upstreamURL.Path = path.Join(target.Path, targetPath)
		upstreamURL.RawQuery = r.URL.RawQuery
		upstreamReq, err := http.NewRequestWithContext(r.Context(), r.Method, upstreamURL.String(), r.Body)
		if err != nil {
			logger.Error("failed to create upstream request", slog.Any("error", err))
			http.Error(w,
				generateErrorClientText(r.Context(), http.StatusInternalServerError),
				http.StatusInternalServerError,
			)
			return
		}
		for header, values := range r.Header {
			for _, value := range values {
				upstreamReq.Header.Add(header, value)
			}
		}
		// Execute it
		upstreamAnswer, err := http.DefaultClient.Do(upstreamReq)
		if err != nil {
			logger.Error("failed to send upstream request", slog.Any("error", err))
			switch {
			case errors.Is(err, syscall.ECONNREFUSED):
				http.Error(w,
					generateErrorClientText(r.Context(), http.StatusBadGateway),
					http.StatusBadGateway,
				)
			default:
				http.Error(w,
					generateErrorClientText(r.Context(), http.StatusInternalServerError),
					http.StatusInternalServerError,
				)
			}
			return
		}
		defer upstreamAnswer.Body.Close()
		// Stream it back
		for header, values := range upstreamAnswer.Header {
			for _, value := range values {
				w.Header().Add(header, value)
			}
		}
		w.WriteHeader(upstreamAnswer.StatusCode)
		if _, err = io.Copy(w, upstreamAnswer.Body); err != nil {
			logger.Error("failed to stream back response", slog.String("error", err.Error()))
		}
	}
}

func generateErrorClientText(ctx context.Context, statusCode int) string {
	return fmt.Sprintf("%s - check qwen3-rp logs for more details (request id #%v)",
		http.StatusText(statusCode),
		ctx.Value(httplog.ReqIDKey),
	)
}

func deepRequestInspection(body io.ReadCloser, mode mode, logger *slog.Logger) (newBody io.ReadCloser, detectedMode mode, err error) {
	// Read the body
	raw, err := io.ReadAll(body)
	if err != nil {
		err = fmt.Errorf("failed to read body: %w", err)
		return
	}
	// Parse the body as JSON
	var data map[string]any
	if err = json.Unmarshal(raw, &data); err != nil {
		err = fmt.Errorf("failed to parse body as JSON: %w", err)
		return
	}
	// Handles messages
	messages, ok := data["messages"]
	if !ok {
		err = errors.New("chat completion body does not contain 'messages' key")
		return
	}
	typedMessages, ok := messages.([]any)
	if !ok {
		err = fmt.Errorf("'messages' key is not a slice: %T", messages)
		return
	}
	if len(typedMessages) == 0 {
		err = errors.New("'messages' slice is empty")
		return
	}
	if detectedMode, err = detector(typedMessages); err != nil {
		err = fmt.Errorf("failed to detect mode by inspecting messages: %w", err)
		return
	}
	switch mode {
	case modeAuto:
		// request came thru the regular endpoint...
		if detectedMode, err = detector(typedMessages); err != nil {
			err = fmt.Errorf("failed to detect mode by inspecting messages: %w", err)
			return
		}
		switch detectedMode {
		case modeAuto:
			// ... and no switches were detected, do nothing
			newBody = io.NopCloser(bytes.NewBuffer(raw))
			return
		case modeThink:
			// ... but an ending thinking switch was detected, update the mode accordingly
			mode = modeThink
		case modeNoThink:
			// ... but an ending no-thinking switch was detected, update the mode accordingly
			mode = modeNoThink
		default:
			err = fmt.Errorf("unknown detected mode: %v", detectedMode)
			return
		}
	case modeThink:
		// request came thru the thinking endpoint...
		switch detectedMode {
		case modeThink:
			// ... and an ending thinking switch was detected, do not edit last message
		case modeAuto:
			// ... but no switches were detected, forcing
			fallthrough
		case modeNoThink:
			// ... but an ending no-thinking switch was detected, forcing
			if err = force(typedMessages, true); err != nil {
				err = fmt.Errorf("failed to force messages for thinking mode: %w", err)
				return
			}
		default:
			err = fmt.Errorf("unknown detected mode: %v", detectedMode)
			return
		}
	case modeNoThink:
		// request came thru the no-thinking endpoint...
		switch detectedMode {
		case modeNoThink:
			// ... and an ending no-thinking switch was detected, do not edit last message
		case modeAuto:
			// ... but no switches were detected, forcing
			fallthrough
		case modeThink:
			// ... but an ending thinking switch was detected, forcing
			if err = force(typedMessages, false); err != nil {
				err = fmt.Errorf("failed to force messages for no-thinking mode: %w", err)
				return
			}
		default:
			err = fmt.Errorf("unknown detected mode: %v", detectedMode)
			return
		}
	default:
		err = fmt.Errorf("unknown mode: %v", mode)
		return
	}
	// Set sampling parameters according to mode
	var temperature, topP float64
	switch mode {
	case modeThink:
		temperature = thinkTemperature
		topP = thinkTopP
	case modeNoThink:
		temperature = noThinkTemperature
		topP = noThinkTopP
	default:
		err = fmt.Errorf("can not set sampling parameters for unknown mode: %v", mode)
		return
	}
	applySamplingParams(data, temperature, topP, logger)
	// Marshal the body back to JSON
	if raw, err = json.Marshal(data); err != nil {
		err = fmt.Errorf("failed to marshal body back to JSON: %w", err)
		return
	}
	newBody = io.NopCloser(bytes.NewBuffer(raw))
	return
}

func detector(messages []any) (detectedMode mode, err error) {
	for i := len(messages) - 1; i >= 0; i-- {
		message, ok := messages[i].(map[string]any)
		if !ok {
			err = fmt.Errorf("message at index %d is not a map: %T", i, messages[i])
			return
		}
		content, ok := message["content"]
		if !ok {
			err = errors.New("last message does not contain 'content' key")
			return
		}
		typedContent, ok := content.(string)
		if !ok {
			err = fmt.Errorf("'content' key is not a string: %T", content)
			return
		}
		switch checkTextSwitch(typedContent) {
		case thinkSwitch:
			detectedMode = modeThink
			return
		case noThinkSwitch:
			detectedMode = modeNoThink
			return
		default:
			// continue searching
		}
	}
	return
}

func checkTextSwitch(input string) string {
	for i := len(input) - 1; i >= 0; i-- {
		for j, suffix := range suffixes {
			if i >= suffixLengths[j]-1 {
				if input[i-suffixLengths[j]+1:i+1] == suffix {
					return suffix
				}
			}
		}
	}
	return ""
}

func force(messages []any, think bool) (err error) {
	firstMessage, ok := messages[len(messages)-1].(map[string]any)
	if !ok {
		err = fmt.Errorf("last message is not a map: %T", messages[len(messages)-1])
		return
	}
	content, ok := firstMessage["content"]
	if !ok {
		err = errors.New("last message does not contain 'content' key")
		return
	}
	typedContent, ok := content.(string)
	if !ok {
		err = fmt.Errorf("'content' key is not a string: %T", content)
		return
	}
	if think {
		typedContent += " " + thinkSwitch
	} else {
		typedContent += " " + noThinkSwitch
	}
	firstMessage["content"] = typedContent
	messages[len(messages)-1] = firstMessage
	return
}

func applySamplingParams(data map[string]any, temperature, topP float64, logger *slog.Logger) {
	// Max Tokens
	if _, exists := data[maxTokensKey]; !exists {
		data[maxTokensKey] = maxTokens
	} else {
		logger.Debug("max_tokens already set in request, not modifying",
			slog.Any("value", data[maxTokensKey]),
			slog.Float64("default_value", maxTokens),
		)
	}
	// Temperature
	if _, exists := data[temperatureKey]; !exists {
		data[temperatureKey] = temperature
	} else {
		logger.Debug("temperature already set in request, not modifying",
			slog.Any("value", data[temperatureKey]),
			slog.Float64("default_value", temperature),
		)
	}
	// Top P
	if _, exists := data[topPKey]; !exists {
		data[topPKey] = topP
	} else {
		logger.Debug("top_p already set in request, not modifying",
			slog.Any("value", data[topPKey]),
			slog.Float64("default_value", topP),
		)
	}
	// Top K
	if _, exists := data[topKKey]; !exists {
		data[topKKey] = TopK
	} else {
		logger.Debug("top_k already set in request, not modifying",
			slog.Any("value", data[topKKey]),
			slog.Int("default_value", TopK),
		)
	}
	// Min P
	if _, exists := data[minPKey]; !exists {
		data[minPKey] = minP
	} else {
		logger.Debug("min_p already set in request, not modifying",
			slog.Any("value", data[minPKey]),
			slog.Float64("default_value", minP),
		)
	}
	// Presence Penalty
	if _, exists := data[presencePenaltyKey]; !exists {
		data[presencePenaltyKey] = presencePenalty
	} else {
		logger.Debug("presence_penalty already set in request, not modifying",
			slog.Any("value", data[presencePenaltyKey]),
			slog.Float64("default_value", presencePenalty),
		)
	}
}
