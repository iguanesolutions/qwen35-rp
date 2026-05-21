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

func tokenize(httpCli *http.Client, target *url.URL,
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

		// Parse as generic map to read only the model field
		var reqData map[string]any
		if err := json.Unmarshal(requestBody, &reqData); err != nil {
			logger.Error("failed to parse body as JSON", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusBadRequest)
			return
		}

		// Act based on the model field
		switch reqData["model"] {
		case "":
			// by default vllm accept a empty model name as it serves only one model
			logger.Debug("tokenize request received without a model name, accept it anyway and set the actual served model name",
				slog.String("served_model", servedModel),
			)
			reqData["model"] = servedModel
		case thinkingGeneral, thinkingCoding, instructGeneral, instructReasoning:
			// user has provided a valid (virtual) model name
			logger.Debug("tokenize request received with a valid virtual model name",
				slog.Any("virtual_model", reqData["model"]),
				slog.String("served_model", servedModel),
			)
			reqData["model"] = servedModel
		default:
			// invalid model name
			logger.Error("tokenize request received with an invalid model name",
				slog.Any("requested_model", reqData["model"]),
				slog.String("served_model", servedModel),
			)
			httpError(ctx, w, http.StatusBadRequest)
			return
		}

		// Marshal the modified request body
		if requestBody, err = json.Marshal(reqData); err != nil {
			logger.Error("failed to marshal modified request body", slog.String("error", err.Error()))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}

		// Create a new request with the modified body
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
		modifiedRequests.Add(1)

		// Copy response as is
		copyHeaders(w, outResp)
		w.WriteHeader(outResp.StatusCode)
		if _, err = io.Copy(w, outResp.Body); err != nil {
			logger.Error("failed to write response", slog.String("error", err.Error()))
		}
	}
}
