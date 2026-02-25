package main

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/url"

	"github.com/hekmon/httplog/v3"
)

// models fetches backend models and enriches with 4 virtual model names
func models(httpCli *http.Client, target *url.URL, servedModel, thinkingGeneral, thinkingCoding, instructGeneral, instructReasoning string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		logger := logger.With(httplog.GetReqIDSLogAttr(ctx))
		logger.Debug("handling /v1/models request")

		// Create request to backend
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, target.String()+"/v1/models", nil)
		if err != nil {
			logger.Error("failed to create models request", slog.Any("error", err))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}

		// Send request to backend
		resp, err := httpCli.Do(req)
		if err != nil {
			logger.Error("failed to fetch models from backend", slog.Any("error", err))
			httpError(ctx, w, http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()

		// Read backend response
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			logger.Error("failed to read models response", slog.Any("error", err))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}

		// Parse JSON response
		var modelsResp map[string]any
		if err := json.Unmarshal(body, &modelsResp); err != nil {
			logger.Error("failed to parse models response", slog.Any("error", err))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}

		// Get original data array
		data, ok := modelsResp["data"].([]any)
		if !ok || len(data) == 0 {
			logger.Warn("no models in backend response, passing through")
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(resp.StatusCode)
			w.Write(body)
			return
		}

		// Find servedModel in backend data array (vLLM can serve multiple models)
		var baseModelMap map[string]any
		found := false
		for _, model := range data {
			modelMap, ok := model.(map[string]any)
			if !ok {
				continue
			}
			modelID, ok := modelMap["id"].(string)
			if !ok {
				continue
			}
			if modelID == servedModel {
				baseModelMap = modelMap
				found = true
				break
			}
		}
		if !found {
			logger.Error("backend is not serving expected model",
				slog.String("expected", servedModel),
				slog.Any("available_models", data),
			)
			httpError(ctx, w, http.StatusBadGateway)
			return
		}
		logger.Debug("backend model found and validated", slog.String("model", servedModel))

		// Virtual model names
		virtualModels := []string{thinkingGeneral, thinkingCoding, instructGeneral, instructReasoning}
		var enrichedData []any

		// Create 4 virtual models
		for _, vmName := range virtualModels {
			// Clone the base model
			vmMap := make(map[string]any)
			for k, v := range baseModelMap {
				vmMap[k] = v
			}
			// Override the id with virtual model name
			vmMap["id"] = vmName
			enrichedData = append(enrichedData, vmMap)
		}

		modelsResp["data"] = enrichedData

		// Marshal enriched response
		enrichedBody, err := json.Marshal(modelsResp)
		if err != nil {
			logger.Error("failed to marshal enriched models response", slog.Any("error", err))
			httpError(ctx, w, http.StatusInternalServerError)
			return
		}

		// Write response
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		w.Write(enrichedBody)
		logger.Info("enriched /v1/models response with 4 virtual models")
	}
}
