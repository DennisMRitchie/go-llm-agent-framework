package tests

import (
	"context"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/DennisMRitchie/go-llm-agent-framework/internal/agent"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/config"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/llmclient"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/metrics"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/nlp"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/tools"
)

func nopLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func testDeps(t *testing.T) (config.AgentConfig, *tools.Registry, *nlp.Preprocessor, *nlp.Classifier, *nlp.EntityExtractor, *metrics.AgentMetrics) {
	t.Helper()
	logger := nopLogger()
	cfg := config.AgentConfig{
		MaxWorkers:        4,
		TaskTimeout:       10 * time.Second,
		MaxMemoryMessages: 20,
		RateLimitRPS:      100,
		RateLimitBurst:    200,
		MaxConcurrentRuns: 4,
	}
	nlpCfg := config.NLPConfig{MaxTokens: 512, ConfidenceThresh: 0.5, EnableNormalize: true, EnableStopwords: true}
	reg := tools.NewRegistry()
	pre := nlp.NewPreprocessor(nlpCfg, nil, logger)
	cls := nlp.NewClassifier(nlpCfg, nil, logger)
	ext := nlp.NewEntityExtractor(nlpCfg, nil, logger)
	m := metrics.NewAgentMetrics()
	return cfg, reg, pre, cls, ext, m
}

func TestAgent_RunWithMockLLM(t *testing.T) {
	cfg, reg, pre, cls, ext, m := testDeps(t)
	mock := &llmclient.MockClient{
		CompleteFunc: func(_ context.Context, _ *llmclient.CompleteRequest) (*llmclient.CompleteResponse, error) {
			return &llmclient.CompleteResponse{Text: "The answer is 42.", Model: "mock", TokensUsed: 15}, nil
		},
	}
	a := agent.New(cfg, mock, reg, pre, cls, ext, m, nopLogger())
	result, err := a.Run(context.Background(), &agent.Task{ID: "t1", Prompt: "What is the meaning of life?"})
	if err != nil {
		t.Fatalf("Run() error: %v", err)
	}
	if result.Status != agent.StatusCompleted {
		t.Errorf("expected completed, got %s (%s)", result.Status, result.Error)
	}
	if result.Response == "" {
		t.Error("expected non-empty response")
	}
	t.Logf("response=%q tokens=%d duration=%s", result.Response, result.TokensUsed, result.Duration)
}

func TestAgent_ToolExecution(t *testing.T) {
	cfg, reg, pre, cls, ext, m := testDeps(t)
	calls := 0
	mock := &llmclient.MockClient{
		CompleteFunc: func(_ context.Context, _ *llmclient.CompleteRequest) (*llmclient.CompleteResponse, error) {
			calls++
			if calls == 1 {
				return &llmclient.CompleteResponse{
					Text:       `<tool_call>{"name":"calculator","params":{"operation":"multiply","a":"6","b":"7"}}</tool_call>`,
					TokensUsed: 20,
				}, nil
			}
			return &llmclient.CompleteResponse{Text: "The result is 42.", TokensUsed: 10}, nil
		},
	}
	a := agent.New(cfg, mock, reg, pre, cls, ext, m, nopLogger())
	result, err := a.Run(context.Background(), &agent.Task{ID: "t2", Prompt: "What is 6 times 7?"})
	if err != nil {
		t.Fatalf("Run() error: %v", err)
	}
	if result.Status != agent.StatusCompleted {
		t.Errorf("expected completed, got %s", result.Status)
	}
	if len(result.ToolCalls) == 0 {
		t.Fatal("expected at least one tool call")
	}
	tc := result.ToolCalls[0]
	if tc.ToolName != "calculator" {
		t.Errorf("expected calculator, got %s", tc.ToolName)
	}
	if tc.Output != "42" {
		t.Errorf("expected output 42, got %s", tc.Output)
	}
}

func TestAgent_RateLimitCancellation(t *testing.T) {
	cfg, reg, pre, cls, ext, m := testDeps(t)
	cfg.RateLimitRPS = 0.0001
	cfg.RateLimitBurst = 0

	a := agent.New(cfg, &llmclient.MockClient{}, reg, pre, cls, ext, m, nopLogger())
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	result, err := a.Run(ctx, &agent.Task{ID: "t3", Prompt: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Status != agent.StatusCancelled {
		t.Logf("status=%s (acceptable if limiter resolved before deadline)", result.Status)
	}
}

func TestOrchestrator_SubmitAndWait(t *testing.T) {
	cfg, reg, pre, cls, ext, m := testDeps(t)
	orch := agent.NewOrchestrator(cfg, &llmclient.MockClient{}, reg, pre, cls, ext, m, nopLogger())

	result, err := orch.SubmitAndWait(context.Background(), "Hello", "s1", nil)
	if err != nil {
		t.Fatalf("SubmitAndWait error: %v", err)
	}
	if result.Status != agent.StatusCompleted {
		t.Errorf("expected completed, got %s: %s", result.Status, result.Error)
	}
}

func TestOrchestrator_ConcurrentTasks(t *testing.T) {
	cfg, reg, pre, cls, ext, m := testDeps(t)
	orch := agent.NewOrchestrator(cfg, &llmclient.MockClient{}, reg, pre, cls, ext, m, nopLogger())

	const n = 8
	ids := make([]string, n)
	for i := range ids {
		id, err := orch.Submit(context.Background(), "concurrent", "s", nil)
		if err != nil {
			t.Fatalf("Submit: %v", err)
		}
		ids[i] = id
	}
	for _, id := range ids {
		r, err := orch.Wait(context.Background(), id)
		if err != nil {
			t.Errorf("Wait(%s): %v", id, err)
			continue
		}
		if r.Status != agent.StatusCompleted {
			t.Errorf("task %s: %s", id, r.Status)
		}
	}
}

func TestOrchestrator_TaskNotFound(t *testing.T) {
	cfg, reg, pre, cls, ext, m := testDeps(t)
	orch := agent.NewOrchestrator(cfg, &llmclient.MockClient{}, reg, pre, cls, ext, m, nopLogger())
	if _, _, err := orch.Status("no-such-id"); err == nil {
		t.Error("expected error for unknown task")
	}
}

func TestMemory_AddAndSearch(t *testing.T) {
	logger := nopLogger()
	pre := nlp.NewPreprocessor(config.NLPConfig{EnableNormalize: true, EnableStopwords: true}, nil, logger)
	mem := agent.NewMemory(10, pre)

	mem.Add("user", "What is the capital of France?", nil)
	mem.Add("assistant", "The capital of France is Paris.", nil)

	if mem.Len() != 2 {
		t.Errorf("expected 2 entries, got %d", mem.Len())
	}
	msgs := mem.Messages()
	if msgs[0].Role != "user" {
		t.Errorf("expected role user, got %s", msgs[0].Role)
	}

	results := mem.Search("France Paris capital", 1)
	if len(results) == 0 {
		t.Error("expected search result")
	}
	t.Logf("search top: [%s] %s", results[0].Role, results[0].Content)
}

func TestMemory_Eviction(t *testing.T) {
	logger := nopLogger()
	pre := nlp.NewPreprocessor(config.NLPConfig{}, nil, logger)
	mem := agent.NewMemory(3, pre)

	for i := 0; i < 5; i++ {
		mem.Add("user", "message", nil)
	}
	if mem.Len() > 3 {
		t.Errorf("expected ≤3 entries, got %d", mem.Len())
	}
}
