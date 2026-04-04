package tests

import (
	"context"
	"testing"
	"time"

	"github.com/DennisMRitchie/go-llm-agent-framework/internal/agent"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/config"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/llmclient"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/metrics"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/nlp"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/tools"
	"go.uber.org/zap/zaptest"
)

func TestAgent_RunTask(t *testing.T) {
	logger := zaptest.NewLogger(t)
	cfg := &config.Config{
		Agent: config.AgentConfig{
			MaxConcurrentTasks: 1,
			Timeout:            time.Minute,
			MaxMemoryMessages:  10,
			RateLimitRPS:       1000.0, // High rate limit for tests
			RateLimitBurst:     1000,
			MaxWorkers:         10,
			TaskTimeout:        time.Minute,
		},
		NLP: config.NLPConfig{
			Enabled: true,
		},
	}

	// Mock LLM client
	llmClient := &mockLLMClient{}

	// Tools registry
	toolReg := tools.NewRegistry()

	// Metrics
	agentMetrics := metrics.NewAgentMetrics()

	// NLP components
	preprocessor := nlp.NewPreprocessor(cfg.NLP, llmClient, logger)
	classifier := nlp.NewClassifier(cfg.NLP, llmClient, logger)
	extractor := nlp.NewEntityExtractor(cfg.NLP, llmClient, logger)

	// Create agent
	ag := agent.New(cfg.Agent, llmClient, toolReg, preprocessor, classifier, extractor, agentMetrics, logger)

	// Test simple calculation
	ctx := context.Background()
	task := &agent.Task{
		ID:        "test-task",
		Prompt:    "What is 2 + 3?",
		SessionID: "test-session",
		CreatedAt: time.Now(),
	}
	result, err := ag.Run(ctx, task)

	if err != nil {
		t.Fatalf("Agent run failed: %v", err)
	}

	if result.Status != agent.StatusCompleted {
		t.Errorf("Expected status completed, got %s", result.Status)
	}

	if result.Response == "" {
		t.Error("Expected non-empty response")
	}
}

// mockLLMClient implements llmclient.Client for testing
type mockLLMClient struct{}

func (m *mockLLMClient) Complete(ctx context.Context, req *llmclient.CompleteRequest) (*llmclient.CompleteResponse, error) {
	return &llmclient.CompleteResponse{
		Text:       "The result is 5.",
		TokensUsed: 10,
	}, nil
}

func (m *mockLLMClient) ClassifyIntent(ctx context.Context, req *llmclient.ClassifyRequest) (*llmclient.ClassifyResponse, error) {
	return &llmclient.ClassifyResponse{
		TopLabel:   req.CandidateLabels[0],
		Confidence: 0.9,
		AllScores:  map[string]float64{req.CandidateLabels[0]: 0.9},
	}, nil
}

func (m *mockLLMClient) ExtractEntities(ctx context.Context, req *llmclient.ExtractRequest) (*llmclient.ExtractResponse, error) {
	return &llmclient.ExtractResponse{Entities: []llmclient.Entity{}}, nil
}
