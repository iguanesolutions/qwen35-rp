package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"path"

	"github.com/hekmon/httplog/v3"
)

// rewriteRequestURL rewrites request URL and Host header for proxying to backend
func rewriteRequestURL(req *http.Request, target *url.URL) {
	req.URL.Scheme = target.Scheme
	req.URL.Host = target.Host
	req.Host = target.Host
	req.URL.Path = path.Join(target.Path, req.URL.Path)
	req.URL.RawPath = ""
	targetQuery := target.RawQuery
	if targetQuery == "" || req.URL.RawQuery == "" {
		req.URL.RawQuery = targetQuery + req.URL.RawQuery
	} else {
		req.URL.RawQuery = targetQuery + "&" + req.URL.RawQuery
	}
}

// maxRequestBodySize is the maximum allowed request body size (100 MB).
// This is a DoS safeguard, not a functional limit — conversation histories
// with base64-encoded images can legitimately be very large.
const maxRequestBodySize = 100 << 20 // 100 MB

// maxSSEEventSize is the maximum allowed size for a single buffered SSE event (10 MB).
// This prevents unbounded memory growth if the backend sends a malformed stream
// without proper event delimiters.
const maxSSEEventSize = 10 << 20 // 10 MB

// hopByHopHeaders lists headers that must not be forwarded by proxies (RFC 7230 §6.1)
var hopByHopHeaders = []string{
	"Connection",
	"Keep-Alive",
	"Proxy-Authenticate",
	"Proxy-Authorization",
	"Te",
	"Trailer",
	"Transfer-Encoding",
	"Upgrade",
}

// stripHopByHopHeaders removes hop-by-hop headers from the request before proxying
func stripHopByHopHeaders(req *http.Request) {
	for _, h := range hopByHopHeaders {
		req.Header.Del(h)
	}
}

// applySamplingParams applies sampling parameters to request data
// If enforce is true, parameters are always set, overriding any client-provided values
func applySamplingParams(data map[string]any, samplingParams map[string]any, logger *slog.Logger, enforce bool) {
	for k, v := range samplingParams {
		if _, ok := data[k]; ok {
			if enforce {
				logger.Debug("enforcing sampling parameter",
					slog.Any("key", k),
					slog.Any("old_value", data[k]),
					slog.Any("new_value", v),
				)
				data[k] = v
			} else {
				logger.Debug("key already set in request, not modifying",
					slog.Any("key", k),
					slog.Any("value", data[k]),
					slog.Any("default_value", v),
				)
			}
			continue
		}
		data[k] = v
	}
}

// copyHeaders copies response headers from backend response to the client ResponseWriter
func copyHeaders(w http.ResponseWriter, resp *http.Response) {
	for header, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(header, value)
		}
	}
}

// httpError writes an OpenAI-compatible JSON error response
func httpError(ctx context.Context, w http.ResponseWriter, statusCode int) {
	reqID := ctx.Value(httplog.ReqIDKey)
	message := fmt.Sprintf("%s - check qwen35-rp logs for more details (request id #%v)",
		http.StatusText(statusCode),
		reqID,
	)
	errResp := map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    http.StatusText(statusCode),
			"code":    statusCode,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	if err := json.NewEncoder(w).Encode(errResp); err != nil {
		logger.Error("failed to write error response", slog.Any("error", err))
	}
}

// readBodyStatusCode returns the appropriate HTTP status code for a body read error.
// Returns 413 for MaxBytesError, 500 for everything else.
func readBodyStatusCode(err error) int {
	var maxBytesErr *http.MaxBytesError
	if errors.As(err, &maxBytesErr) {
		return http.StatusRequestEntityTooLarge
	}
	return http.StatusInternalServerError
}

// extractSSEDataJSON extracts the JSON payload from an SSE event.
// It handles multi-line events by scanning all lines for the first "data: " line
// that contains parseable JSON (skipping [DONE] and empty data lines).
// Returns nil if no valid JSON data line is found.
func extractSSEDataJSON(event []byte) []byte {
	trimmed := bytes.TrimRight(event, "\n\r ")
	lines := bytes.Split(trimmed, []byte("\n"))
	for _, line := range lines {
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		jsonPart := bytes.TrimSpace(line[6:])
		if len(jsonPart) == 0 || bytes.Equal(jsonPart, []byte("[DONE]")) {
			continue
		}
		return jsonPart
	}
	return nil
}
