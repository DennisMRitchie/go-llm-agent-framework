package llmclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"go.uber.org/zap"
)

// Client interface for LLM backend communication
type Client interface {
	ExtractEntities(ctx context.Context, req *ExtractRequest) (*ExtractResponse, error)
	ClassifyIntent(ctx context.Context, req *ClassifyRequest) (*ClassifyResponse, error)
	Complete(ctx context.Context, req *CompleteRequest) (*CompleteResponse, error)
}

// HTTPClient handles HTTP communication with LLM backends
type HTTPClient struct {
	baseURL string
	apiKey  string
	client  *http.Client
	logger  *zap.Logger
}

// NewHTTPClient creates a new HTTP client for LLM backend communication
func NewHTTPClient(cfg LLMBackendConfig, logger *zap.Logger) *HTTPClient {
	return &HTTPClient{
		baseURL: cfg.BaseURL,
		apiKey:  cfg.APIKey,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		logger: logger,
	}
}

// LLMBackendConfig holds configuration for LLM backend
type LLMBackendConfig struct {
	BaseURL string
	APIKey  string
}

// Request represents a request to the LLM
type Request struct {
	Prompt string `json:"prompt"`
	// Add other fields as needed
}

// Response represents a response from the LLM
type Response struct {
	Text string `json:"text"`
	// Add other fields as needed
}

// Entity represents an extracted entity
type Entity struct {
	Text       string  `json:"text"`
	Label      string  `json:"label"`
	Start      int     `json:"start"`
	End        int     `json:"end"`
	Confidence float64 `json:"confidence"`
}

// ExtractRequest represents a request to extract entities
type ExtractRequest struct {
	Text string `json:"text"`
}

// ExtractResponse represents a response from entity extraction
type ExtractResponse struct {
	Entities []Entity `json:"entities"`
}

// ClassifyRequest represents a request to classify intent
type ClassifyRequest struct {
	Text            string   `json:"text"`
	CandidateLabels []string `json:"candidate_labels"`
}

// ClassifyResponse represents a response from intent classification
type ClassifyResponse struct {
	TopLabel   string             `json:"top_label"`
	Confidence float64            `json:"confidence"`
	AllScores  map[string]float64 `json:"all_scores"`
}

// Message represents a chat message
type Message struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	Timestamp int64  `json:"timestamp,omitempty"`
}

// CompleteRequest represents a request to complete a conversation
type CompleteRequest struct {
	Messages  []Message `json:"messages"`
	Prompt    string    `json:"prompt,omitempty"`
	History   []Message `json:"history,omitempty"`
	SessionID string    `json:"session_id,omitempty"`
}

// CompleteResponse represents a response from completion
type CompleteResponse struct {
	Text       string `json:"text"`
	TokensUsed int    `json:"tokens_used"`
}

// Generate sends a prompt to the LLM and returns the response
func (c *HTTPClient) Generate(prompt string) (string, error) {
	req := Request{Prompt: prompt}
	reqBody, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", c.baseURL+"/generate", bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var llmResp Response
	if err := json.NewDecoder(resp.Body).Decode(&llmResp); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}

	return llmResp.Text, nil
}

// ExtractEntities extracts entities from text using the LLM backend
func (c *HTTPClient) ExtractEntities(ctx context.Context, req *ExtractRequest) (*ExtractResponse, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/extract-entities", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var extractResp ExtractResponse
	if err := json.NewDecoder(resp.Body).Decode(&extractResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &extractResp, nil
}

// ClassifyIntent classifies the intent of text using the LLM backend
func (c *HTTPClient) ClassifyIntent(ctx context.Context, req *ClassifyRequest) (*ClassifyResponse, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/classify-intent", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var classifyResp ClassifyResponse
	if err := json.NewDecoder(resp.Body).Decode(&classifyResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &classifyResp, nil
}

// Complete completes a conversation using the LLM backend
func (c *HTTPClient) Complete(ctx context.Context, req *CompleteRequest) (*CompleteResponse, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/complete", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var completeResp CompleteResponse
	if err := json.NewDecoder(resp.Body).Decode(&completeResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &completeResp, nil
}
