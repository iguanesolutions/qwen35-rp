package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/hekmon/httplog/v2"
)

var (
	thinkSamplingParams = map[string]any{
		"temperature":        0.6,
		"top_p":              0.95,
		"top_k":              20,
		"min_p":              0.0,
		"presence_penalty":   0.0,
		"repetition_penalty": 1.0,
	}
	noThinkSamplingParams = map[string]any{
		"temperature":        0.7,
		"top_p":              0.8,
		"top_k":              20,
		"min_p":              0.0,
		"presence_penalty":   1.5,
		"repetition_penalty": 1.0,
	}
)

func proxy(target *url.URL,
	servedModel, thinkingModel, noThinkingModel string) http.HandlerFunc {
	// pooled client
	httpCli := &http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout:   30 * time.Second,
				KeepAlive: 30 * time.Second,
				DualStack: true,
			}).DialContext,
			MaxIdleConns:          100,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
			ForceAttemptHTTP2:     true,
			MaxIdleConnsPerHost:   runtime.GOMAXPROCS(0) + 1,
		},
	}
	return func(w http.ResponseWriter, r *http.Request) {
		logger := logger.With(httplog.GetReqIDSLogAttr(r.Context()))
		logger.Info("received a request",
			slog.String("remote_addr", r.RemoteAddr),
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
		)
		ctx := r.Context()
		requestBody, err := io.ReadAll(r.Body)
		if err != nil {
			logger.Error("failed to read body", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}
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
		// check thinking mode based on model name and apply sampling parameters
		var think bool
		switch modelName {
		case thinkingModel:
			think = true
			applySamplingParams(data, thinkSamplingParams, logger)
		case noThinkingModel:
			think = false
			applySamplingParams(data, noThinkSamplingParams, logger)
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

		requestBody, err = json.Marshal(data)
		if err != nil {
			logger.Error("failed to marshal request body", slog.Any("error", err))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}

		logger.Debug("rewrited request body", slog.String("body", string(requestBody)))

		// outgoing request
		outreq := r.Clone(r.Context())
		rewriteRequestURL(outreq, target)
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
		for header, values := range outResp.Header {
			for _, value := range values {
				w.Header().Add(header, value)
			}
		}
		w.WriteHeader(outResp.StatusCode)
		if _, err = io.Copy(w, outResp.Body); err != nil {
			logger.Error("failed to stream back response", slog.String("error", err.Error()))
		}
	}
}

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

func httpError(ctx context.Context, w http.ResponseWriter, statusCode int) {
	http.Error(w,
		generateErrorClientText(ctx, statusCode),
		statusCode,
	)
}

func generateErrorClientText(ctx context.Context, statusCode int) string {
	return fmt.Sprintf("%s - check qwen35-rp logs for more details (request id #%v)",
		http.StatusText(statusCode),
		ctx.Value(httplog.ReqIDKey),
	)
}
