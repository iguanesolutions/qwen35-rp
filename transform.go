package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"syscall"

	"github.com/hekmon/httplog/v3"
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

func proxy(httpCli *http.Client, target *url.URL,
	servedModel, thinkingModel, noThinkingModel string, enforceSamplingParams bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
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
		// check thinking mode based on model name and apply sampling parameters
		var think bool
		switch modelName {
		case thinkingModel:
			think = true
			applySamplingParams(data, thinkSamplingParams, logger, enforceSamplingParams)
		case noThinkingModel:
			think = false
			applySamplingParams(data, noThinkSamplingParams, logger, enforceSamplingParams)
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
