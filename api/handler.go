// Package api implements the HTTP REST API using only stdlib net/http.
package api

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/DennisMRitchie/go-llm-agent-framework/internal/agent"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/metrics"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/nlp"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/tools"
)

type Handler struct {
	orchestrator *agent.Orchestrator
	toolReg      *tools.Registry
	preprocessor *nlp.Preprocessor
	classifier   *nlp.Classifier
	extractor    *nlp.EntityExtractor
	httpMetrics  *metrics.HTTPMetrics
	logger       *slog.Logger
}

func NewHandler(
	orchestrator *agent.Orchestrator,
	toolReg *tools.Registry,
	preprocessor *nlp.Preprocessor,
	classifier *nlp.Classifier,
	extractor *nlp.EntityExtractor,
	httpMetrics *metrics.HTTPMetrics,
	logger *slog.Logger,
) *Handler {
	return &Handler{
		orchestrator: orchestrator,
		toolReg:      toolReg,
		preprocessor: preprocessor,
		classifier:   classifier,
		extractor:    extractor,
		httpMetrics:  httpMetrics,
		logger:       logger,
	}
}

// Register wires all routes on the given mux.
func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /health", h.Health)
	mux.HandleFunc("GET /metrics", h.Metrics)

	mux.HandleFunc("POST /api/v1/agent/run", h.RunTask)
	mux.HandleFunc("POST /api/v1/agent/run/sync", h.RunTaskSync)
	mux.HandleFunc("GET /api/v1/agent/tasks", h.ListTasks)
	mux.HandleFunc("GET /api/v1/agent/tasks/{id}", h.GetTask)
	mux.HandleFunc("DELETE /api/v1/agent/tasks/{id}", h.CancelTask)

	mux.HandleFunc("POST /api/v1/nlp/process", h.ProcessText)
	mux.HandleFunc("POST /api/v1/nlp/classify", h.ClassifyIntent)
	mux.HandleFunc("POST /api/v1/nlp/entities", h.ExtractEntities)
	mux.HandleFunc("POST /api/v1/nlp/keywords", h.ExtractKeywords)

	mux.HandleFunc("GET /api/v1/tools", h.ListTools)
}

// --- Helpers ---

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func readJSON(r *http.Request, v any) error {
	defer r.Body.Close()
	return json.NewDecoder(r.Body).Decode(v)
}

func errJSON(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

// --- Handlers ---

func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status":  "ok",
		"service": "go-llm-agent-framework",
		"time":    time.Now().UTC(),
	})
}

func (h *Handler) Metrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	metrics.WriteMetrics(w)
}

func (h *Handler) RunTask(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Prompt    string            `json:"prompt"`
		SessionID string            `json:"session_id"`
		Context   map[string]string `json:"context"`
	}
	if err := readJSON(r, &req); err != nil || req.Prompt == "" {
		errJSON(w, http.StatusBadRequest, "prompt is required")
		return
	}
	taskID, err := h.orchestrator.Submit(r.Context(), req.Prompt, req.SessionID, req.Context)
	if err != nil {
		h.logger.Error("submit task failed", "error", err)
		errJSON(w, http.StatusInternalServerError, "failed to submit task")
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{
		"task_id": taskID, "status": "pending", "created_at": time.Now().UTC(),
	})
}

func (h *Handler) RunTaskSync(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Prompt    string            `json:"prompt"`
		SessionID string            `json:"session_id"`
		Context   map[string]string `json:"context"`
	}
	if err := readJSON(r, &req); err != nil || req.Prompt == "" {
		errJSON(w, http.StatusBadRequest, "prompt is required")
		return
	}
	result, err := h.orchestrator.SubmitAndWait(r.Context(), req.Prompt, req.SessionID, req.Context)
	if err != nil {
		errJSON(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, taskResultJSON(result))
}

func (h *Handler) GetTask(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	status, result, err := h.orchestrator.Status(id)
	if err != nil {
		errJSON(w, http.StatusNotFound, err.Error())
		return
	}
	if result != nil {
		writeJSON(w, http.StatusOK, taskResultJSON(result))
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"task_id": id, "status": string(status)})
}

func (h *Handler) ListTasks(w http.ResponseWriter, r *http.Request) {
	results := h.orchestrator.ListTasks()
	writeJSON(w, http.StatusOK, map[string]any{"tasks": results, "count": len(results)})
}

func (h *Handler) CancelTask(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if err := h.orchestrator.Cancel(id); err != nil {
		errJSON(w, http.StatusNotFound, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"task_id": id, "status": "cancellation_requested"})
}

func (h *Handler) ProcessText(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Text string `json:"text"`
	}
	if err := readJSON(r, &req); err != nil || req.Text == "" {
		errJSON(w, http.StatusBadRequest, "text is required")
		return
	}
	res := h.preprocessor.Process(req.Text)
	writeJSON(w, http.StatusOK, map[string]any{
		"original": res.Original, "normalized": res.Normalized,
		"word_count": res.WordCount, "char_count": res.CharCount,
		"language": res.Language, "sentences": res.Sentences,
		"token_count": len(res.Tokens),
	})
}

func (h *Handler) ClassifyIntent(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Text   string   `json:"text"`
		Labels []string `json:"labels"`
	}
	if err := readJSON(r, &req); err != nil || req.Text == "" || len(req.Labels) == 0 {
		errJSON(w, http.StatusBadRequest, "text and labels are required")
		return
	}
	res, err := h.classifier.Classify(r.Context(), req.Text, req.Labels)
	if err != nil {
		errJSON(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"intent": res.Intent, "confidence": res.Confidence, "all_scores": res.AllScores,
	})
}

func (h *Handler) ExtractEntities(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Text string `json:"text"`
	}
	if err := readJSON(r, &req); err != nil || req.Text == "" {
		errJSON(w, http.StatusBadRequest, "text is required")
		return
	}
	entities, err := h.extractor.Extract(r.Context(), req.Text)
	if err != nil {
		errJSON(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"entities": entities, "count": len(entities)})
}

func (h *Handler) ExtractKeywords(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Text string `json:"text"`
		TopN int    `json:"top_n"`
	}
	if err := readJSON(r, &req); err != nil || req.Text == "" {
		errJSON(w, http.StatusBadRequest, "text is required")
		return
	}
	if req.TopN <= 0 {
		req.TopN = 10
	}
	writeJSON(w, http.StatusOK, map[string]any{"keywords": h.preprocessor.Keywords(req.Text, req.TopN)})
}

func (h *Handler) ListTools(w http.ResponseWriter, r *http.Request) {
	schemas := h.toolReg.ToolSchemas()
	writeJSON(w, http.StatusOK, map[string]any{"tools": schemas, "count": len(schemas)})
}

// --- Middleware ---

func MetricsMiddleware(m *metrics.HTTPMetrics, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		m.ActiveRequests.Inc()
		defer m.ActiveRequests.Dec()
		rw := &responseWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(rw, r)
		dur := time.Since(start)
		m.RequestsTotal.With(r.Method, r.URL.Path, strconv.Itoa(rw.status)).Inc()
		m.RequestDuration.With(r.Method, r.URL.Path).Observe(dur.Seconds())
	})
}

func RequestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" {
			id = strconv.FormatInt(time.Now().UnixNano(), 36)
		}
		w.Header().Set("X-Request-ID", id)
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), ctxKeyRequestID{}, id)))
	})
}

type ctxKeyRequestID struct{}

func SlogLogger(logger *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &responseWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(rw, r)
		logger.Info("http",
			"method", r.Method, "path", r.URL.Path,
			"status", rw.status, "duration_ms", time.Since(start).Milliseconds(),
			"ip", strings.Split(r.RemoteAddr, ":")[0],
		)
	})
}

type responseWriter struct {
	http.ResponseWriter
	status int
}

func (rw *responseWriter) WriteHeader(s int) {
	rw.status = s
	rw.ResponseWriter.WriteHeader(s)
}

// --- Response helpers ---

func taskResultJSON(r *agent.TaskResult) map[string]any {
	return map[string]any{
		"task_id":     r.TaskID,
		"status":      string(r.Status),
		"response":    r.Response,
		"tool_calls":  r.ToolCalls,
		"tokens_used": r.TokensUsed,
		"duration_ms": r.Duration.Milliseconds(),
		"error":       r.Error,
	}
}
