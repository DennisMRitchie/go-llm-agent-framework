package llmclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/DennisMRitchie/go-llm-agent-framework/internal/config"
)

type Message struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	Timestamp int64  `json:"timestamp_ms,omitempty"`
}

type CompleteRequest struct {
	Prompt     string            `json:"prompt"`
	History    []Message         `json:"history,omitempty"`
	Parameters map[string]string `json:"parameters,omitempty"`
	SessionID  string            `json:"session_id,omitempty"`
}

type CompleteResponse struct {
	Text       string            `json:"text"`
	Model      string            `json:"model"`
	TokensUsed int               `json:"tokens_used"`
	Metadata   map[string]string `json:"metadata,omitempty"`
	LatencyMs  float64           `json:"latency_ms"`
}

type ClassifyRequest struct {
	Text            string   `json:"text"`
	CandidateLabels []string `json:"candidate_labels"`
	MultiLabel      bool     `json:"multi_label,omitempty"`
}

type ClassifyResponse struct {
	TopLabel   string             `json:"top_label"`
	Confidence float64            `json:"confidence"`
	AllScores  map[string]float64 `json:"all_scores,omitempty"`
}

type ExtractRequest struct {
	Text        string   `json:"text"`
	EntityTypes []string `json:"entity_types,omitempty"`
}

type Entity struct {
	Text       string  `json:"text"`
	Label      string  `json:"label"`
	Start      int     `json:"start"`
	End        int     `json:"end"`
	Confidence float64 `json:"confidence"`
}

type ExtractResponse struct {
	Entities []Entity `json:"entities"`
}

type Client interface {
	Complete(ctx context.Context, req *CompleteRequest) (*CompleteResponse, error)
	ClassifyIntent(ctx context.Context, req *ClassifyRequest) (*ClassifyResponse, error)
	ExtractEntities(ctx context.Context, req *ExtractRequest) (*ExtractResponse, error)
	Ping(ctx context.Context) error
}

type httpClient struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	maxRetries int
	logger     *slog.Logger
}

func NewHTTPClient(cfg config.LLMBackendConfig, logger *slog.Logger) Client {
	return &httpClient{
		baseURL:    cfg.BaseURL,
		apiKey:     cfg.APIKey,
		httpClient: &http.Client{Timeout: cfg.Timeout},
		maxRetries: cfg.MaxRetries,
		logger:     logger,
	}
}

func (c *httpClient) Complete(ctx context.Context, req *CompleteRequest) (*CompleteResponse, error) {
	var resp CompleteResponse
	if err := c.post(ctx, "/v1/complete", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (c *httpClient) ClassifyIntent(ctx context.Context, req *ClassifyRequest) (*ClassifyResponse, error) {
	var resp ClassifyResponse
	if err := c.post(ctx, "/v1/classify", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (c *httpClient) ExtractEntities(ctx context.Context, req *ExtractRequest) (*ExtractResponse, error) {
	var resp ExtractResponse
	if err := c.post(ctx, "/v1/entities", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (c *httpClient) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ping failed: status %d", resp.StatusCode)
	}
	return nil
}

func (c *httpClient) post(ctx context.Context, path string, body, out any) error {
	data, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(time.Duration(attempt*attempt) * 100 * time.Millisecond):
			}
			c.logger.Warn("retrying LLM request", "attempt", attempt, "error", lastErr)
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(data))
		if err != nil {
			return fmt.Errorf("build request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		if c.apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+c.apiKey)
		}

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("http do: %w", err)
			continue
		}

		raw, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = fmt.Errorf("read body: %w", err)
			continue
		}

		if resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("backend error %d: %s", resp.StatusCode, string(raw))
			continue
		}
		if resp.StatusCode >= 400 {
			return fmt.Errorf("client error %d: %s", resp.StatusCode, string(raw))
		}

		if err := json.Unmarshal(raw, out); err != nil {
			return fmt.Errorf("unmarshal response: %w", err)
		}
		return nil
	}
	return fmt.Errorf("max retries exceeded: %w", lastErr)
}

// MockClient for testing.
type MockClient struct {
	CompleteFunc        func(ctx context.Context, req *CompleteRequest) (*CompleteResponse, error)
	ClassifyIntentFunc  func(ctx context.Context, req *ClassifyRequest) (*ClassifyResponse, error)
	ExtractEntitiesFunc func(ctx context.Context, req *ExtractRequest) (*ExtractResponse, error)
}

func (m *MockClient) Complete(ctx context.Context, req *CompleteRequest) (*CompleteResponse, error) {
	if m.CompleteFunc != nil {
		return m.CompleteFunc(ctx, req)
	}
	return &CompleteResponse{Text: "mock response for: " + req.Prompt, Model: "mock", TokensUsed: 10}, nil
}

func (m *MockClient) ClassifyIntent(ctx context.Context, req *ClassifyRequest) (*ClassifyResponse, error) {
	if m.ClassifyIntentFunc != nil {
		return m.ClassifyIntentFunc(ctx, req)
	}
	label := "unknown"
	if len(req.CandidateLabels) > 0 {
		label = req.CandidateLabels[0]
	}
	return &ClassifyResponse{TopLabel: label, Confidence: 0.9}, nil
}

func (m *MockClient) ExtractEntities(ctx context.Context, req *ExtractRequest) (*ExtractResponse, error) {
	if m.ExtractEntitiesFunc != nil {
		return m.ExtractEntitiesFunc(ctx, req)
	}
	return &ExtractResponse{}, nil
}

func (m *MockClient) Ping(_ context.Context) error { return nil }
