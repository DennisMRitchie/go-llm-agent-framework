// Package api contains the HTTP handlers for the agent framework REST API.
package api

import (
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"

	"github.com/DennisMRitchie/go-llm-agent-framework/internal/agent"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/metrics"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/nlp"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/tools"
)

// Handler holds all API dependencies.
type Handler struct {
	orchestrator *agent.Orchestrator
	toolReg      *tools.Registry
	preprocessor *nlp.Preprocessor
	classifier   *nlp.Classifier
	extractor    *nlp.EntityExtractor
	httpMetrics  *metrics.HTTPMetrics
	logger       *zap.Logger
}

// NewHandler constructs the API handler.
func NewHandler(
	orchestrator *agent.Orchestrator,
	toolReg *tools.Registry,
	preprocessor *nlp.Preprocessor,
	classifier *nlp.Classifier,
	extractor *nlp.EntityExtractor,
	httpMetrics *metrics.HTTPMetrics,
	logger *zap.Logger,
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

// Register wires all routes onto the given engine.
func (h *Handler) Register(r *gin.Engine) {
	// Observability
	r.GET("/health", h.Health)
	r.GET("/metrics", gin.WrapH(promhttp.Handler()))

	v1 := r.Group("/api/v1")
	{
		// Agent tasks
		v1.POST("/agent/run", h.RunTask)
		v1.POST("/agent/run/sync", h.RunTaskSync)
		v1.GET("/agent/tasks", h.ListTasks)
		v1.GET("/agent/tasks/:id", h.GetTask)
		v1.DELETE("/agent/tasks/:id", h.CancelTask)

		// NLP endpoints
		v1.POST("/nlp/process", h.ProcessText)
		v1.POST("/nlp/classify", h.ClassifyIntent)
		v1.POST("/nlp/entities", h.ExtractEntities)
		v1.POST("/nlp/keywords", h.ExtractKeywords)

		// Tools registry
		v1.GET("/tools", h.ListTools)
	}
}

// --- Request/Response types ---

type RunTaskRequest struct {
	Prompt    string            `json:"prompt" binding:"required"`
	SessionID string            `json:"session_id"`
	Context   map[string]string `json:"context"`
}

type RunTaskResponse struct {
	TaskID    string    `json:"task_id"`
	Status    string    `json:"status"`
	CreatedAt time.Time `json:"created_at"`
}

type TaskResponse struct {
	TaskID     string                   `json:"task_id"`
	Status     string                   `json:"status"`
	Response   string                   `json:"response,omitempty"`
	ToolCalls  []agent.ToolCallRecord   `json:"tool_calls,omitempty"`
	TokensUsed int                      `json:"tokens_used,omitempty"`
	DurationMs int64                    `json:"duration_ms,omitempty"`
	Error      string                   `json:"error,omitempty"`
}

type ProcessTextRequest struct {
	Text string `json:"text" binding:"required"`
}

type ClassifyRequest struct {
	Text   string   `json:"text" binding:"required"`
	Labels []string `json:"labels" binding:"required"`
}

type ExtractEntitiesRequest struct {
	Text string `json:"text" binding:"required"`
}

type KeywordsRequest struct {
	Text  string `json:"text" binding:"required"`
	TopN  int    `json:"top_n"`
}

// --- Handlers ---

func (h *Handler) Health(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":  "ok",
		"service": "go-llm-agent-framework",
		"time":    time.Now().UTC(),
	})
}

// RunTask submits a task asynchronously and returns the task ID.
func (h *Handler) RunTask(c *gin.Context) {
	var req RunTaskRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	taskID, err := h.orchestrator.Submit(c.Request.Context(), req.Prompt, req.SessionID, req.Context)
	if err != nil {
		h.logger.Error("submit task failed", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to submit task"})
		return
	}

	c.JSON(http.StatusAccepted, RunTaskResponse{
		TaskID:    taskID,
		Status:    string(agent.StatusPending),
		CreatedAt: time.Now().UTC(),
	})
}

// RunTaskSync submits a task and waits for the result (blocking).
func (h *Handler) RunTaskSync(c *gin.Context) {
	var req RunTaskRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	result, err := h.orchestrator.SubmitAndWait(c.Request.Context(), req.Prompt, req.SessionID, req.Context)
	if err != nil {
		h.logger.Error("sync task failed", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, toTaskResponse(result))
}

// GetTask returns the status and result of a task.
func (h *Handler) GetTask(c *gin.Context) {
	id := c.Param("id")
	status, result, err := h.orchestrator.Status(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	if result != nil {
		c.JSON(http.StatusOK, toTaskResponse(result))
		return
	}
	c.JSON(http.StatusOK, gin.H{"task_id": id, "status": string(status)})
}

// ListTasks returns all tracked tasks.
func (h *Handler) ListTasks(c *gin.Context) {
	results := h.orchestrator.ListTasks()
	c.JSON(http.StatusOK, gin.H{
		"tasks": results,
		"count": len(results),
	})
}

// CancelTask cancels a running task.
func (h *Handler) CancelTask(c *gin.Context) {
	id := c.Param("id")
	if err := h.orchestrator.Cancel(id); err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"task_id": id, "status": "cancellation_requested"})
}

// ProcessText runs the NLP preprocessing pipeline.
func (h *Handler) ProcessText(c *gin.Context) {
	var req ProcessTextRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	result := h.preprocessor.Process(req.Text)
	c.JSON(http.StatusOK, gin.H{
		"original":   result.Original,
		"normalized": result.Normalized,
		"word_count": result.WordCount,
		"char_count": result.CharCount,
		"language":   result.Language,
		"sentences":  result.Sentences,
		"token_count": len(result.Tokens),
	})
}

// ClassifyIntent classifies text into provided labels.
func (h *Handler) ClassifyIntent(c *gin.Context) {
	var req ClassifyRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	result, err := h.classifier.Classify(c.Request.Context(), req.Text, req.Labels)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"intent":     result.Intent,
		"confidence": result.Confidence,
		"all_scores": result.AllScores,
	})
}

// ExtractEntities extracts named entities from text.
func (h *Handler) ExtractEntities(c *gin.Context) {
	var req ExtractEntitiesRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	entities, err := h.extractor.Extract(c.Request.Context(), req.Text)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"entities": entities,
		"count":    len(entities),
	})
}

// ExtractKeywords returns the top N keywords from text.
func (h *Handler) ExtractKeywords(c *gin.Context) {
	var req KeywordsRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if req.TopN <= 0 {
		req.TopN = 10
	}
	keywords := h.preprocessor.Keywords(req.Text, req.TopN)
	c.JSON(http.StatusOK, gin.H{"keywords": keywords})
}

// ListTools returns all registered tools with their schemas.
func (h *Handler) ListTools(c *gin.Context) {
	schemas := h.toolReg.ToolSchemas()
	c.JSON(http.StatusOK, gin.H{
		"tools": schemas,
		"count": len(schemas),
	})
}

// --- Middleware ---

// MetricsMiddleware records HTTP metrics.
func MetricsMiddleware(m *metrics.HTTPMetrics) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		m.ActiveRequests.Inc()
		defer m.ActiveRequests.Dec()

		c.Next()

		duration := time.Since(start)
		status := strconv.Itoa(c.Writer.Status())
		path := c.FullPath()
		if path == "" {
			path = "unknown"
		}

		m.RequestsTotal.WithLabelValues(c.Request.Method, path, status).Inc()
		m.RequestDuration.WithLabelValues(c.Request.Method, path).Observe(duration.Seconds())
	}
}

// RequestIDMiddleware injects a unique request ID.
func RequestIDMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		id := c.GetHeader("X-Request-ID")
		if id == "" {
			// Generate a short ID from time
			id = strconv.FormatInt(time.Now().UnixNano(), 36)
		}
		c.Set("request_id", id)
		c.Header("X-Request-ID", id)
		c.Next()
	}
}

// ZapLogger returns a Gin middleware that logs requests with zap.
func ZapLogger(logger *zap.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()
		logger.Info("http request",
			zap.String("method", c.Request.Method),
			zap.String("path", c.Request.URL.Path),
			zap.Int("status", c.Writer.Status()),
			zap.Duration("duration", time.Since(start)),
			zap.String("ip", c.ClientIP()),
			zap.String("request_id", c.GetString("request_id")),
		)
	}
}

// --- Helpers ---

func toTaskResponse(r *agent.TaskResult) TaskResponse {
	resp := TaskResponse{
		TaskID:     r.TaskID,
		Status:     string(r.Status),
		Response:   r.Response,
		ToolCalls:  r.ToolCalls,
		TokensUsed: r.TokensUsed,
		DurationMs: r.Duration.Milliseconds(),
		Error:      r.Error,
	}
	return resp
}
