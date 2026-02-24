package main

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"log/slog"

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

// applySamplingParams applies sampling parameters to request data
func applySamplingParams(data map[string]any, samplingParams map[string]any, logger *slog.Logger) {
	for k, v := range samplingParams {
		if _, ok := data[k]; ok {
			logger.Debug("key already set in request, not modifying",
				slog.Any("key", k),
				slog.Any("value", data[k]),
				slog.Any("default_value", v),
			)
			continue
		}
		data[k] = v
	}
}

// httpError writes HTTP error response
func httpError(ctx context.Context, w http.ResponseWriter, statusCode int) {
	http.Error(w,
		generateErrorClientText(ctx, statusCode),
		statusCode,
	)
}

// generateErrorClientText generates error text with request ID
func generateErrorClientText(ctx context.Context, statusCode int) string {
	return fmt.Sprintf("%s - check qwen35-rp logs for more details (request id #%v)",
		http.StatusText(statusCode),
		ctx.Value(httplog.ReqIDKey),
	)
}
