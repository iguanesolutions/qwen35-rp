package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strings"

	"github.com/hekmon/httplog/v3"
)

// singleJoiningSlash joins two path segments with proper slash handling
func singleJoiningSlash(a, b string) string {
	aslash := strings.HasSuffix(a, "/")
	bslash := strings.HasPrefix(b, "/")
	switch {
	case aslash && bslash:
		return a + b[1:]
	case !aslash && !bslash:
		return a + "/" + b
	}
	return a + b
}

// joinURLPath joins URL paths from two URLs
func joinURLPath(a, b *url.URL) (path, rawpath string) {
	if a.RawPath == "" && b.RawPath == "" {
		return singleJoiningSlash(a.Path, b.Path), ""
	}
	// Same as singleJoiningSlash, but uses EscapedPath to determine
	// whether a slash should be added
	apath := a.EscapedPath()
	bpath := b.EscapedPath()

	aslash := strings.HasSuffix(apath, "/")
	bslash := strings.HasPrefix(bpath, "/")

	switch {
	case aslash && bslash:
		return a.Path + b.Path[1:], apath + bpath[1:]
	case !aslash && !bslash:
		return a.Path + "/" + b.Path, apath + "/" + bpath
	}
	return a.Path + b.Path, apath + bpath
}

// rewriteRequestURL rewrites request URL for proxying to backend
func rewriteRequestURL(req *http.Request, target *url.URL) {
	targetQuery := target.RawQuery
	req.URL.Scheme = target.Scheme
	req.URL.Host = target.Host
	req.URL.Path, req.URL.RawPath = joinURLPath(target, req.URL)
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
	json.NewEncoder(w).Encode(errResp)
}
