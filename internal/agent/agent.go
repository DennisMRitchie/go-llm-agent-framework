package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/DennisMRitchie/go-llm-agent-framework/internal/config"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/llmclient"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/metrics"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/nlp"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/tools"
)

type TaskStatus string

const (
	StatusPending   TaskStatus = "pending"
	StatusRunning   TaskStatus = "running"
	StatusCompleted TaskStatus = "completed"
	StatusFailed    TaskStatus = "failed"
	StatusCancelled TaskStatus = "cancelled"
)

type Task struct {
	ID        string
	Prompt    string
	SessionID string
	Context   map[string]string
	CreatedAt time.Time
}

type TaskResult struct {
	TaskID      string
	Status      TaskStatus
	Response    string
	ToolCalls   []ToolCallRecord
	TokensUsed  int
	Duration    time.Duration
	Error       string
	CompletedAt time.Time
}

type ToolCallRecord struct {
	ToolName string
	Params   map[string]string
	Output   string
	IsError  bool
	Duration time.Duration
}

// rateLimiter is a simple token-bucket rate limiter (no external deps).
type rateLimiter struct {
	tokens   float64
	maxBurst float64
	rps      float64
	last     time.Time
	mu       sync.Mutex
}

func newRateLimiter(rps float64, burst int) *rateLimiter {
	return &rateLimiter{tokens: float64(burst), maxBurst: float64(burst), rps: rps, last: time.Now()}
}

func (r *rateLimiter) Wait(ctx context.Context) error {
	for {
		r.mu.Lock()
		now := time.Now()
		r.tokens += r.rps * now.Sub(r.last).Seconds()
		if r.tokens > r.maxBurst {
			r.tokens = r.maxBurst
		}
		r.last = now
		if r.tokens >= 1 {
			r.tokens--
			r.mu.Unlock()
			return nil
		}
		wait := time.Duration(float64(time.Second) / r.rps)
		r.mu.Unlock()

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(wait):
		}
	}
}

type Agent struct {
	id         string
	cfg        config.AgentConfig
	llmClient  llmclient.Client
	toolReg    *tools.Registry
	memory     *Memory
	nlpProc    *nlp.Preprocessor
	classifier *nlp.Classifier
	extractor  *nlp.EntityExtractor
	metrics    *metrics.AgentMetrics
	limiter    *rateLimiter
	logger     *slog.Logger
	workerSem  chan struct{}
}

func New(
	cfg config.AgentConfig,
	llmClient llmclient.Client,
	toolReg *tools.Registry,
	preprocessor *nlp.Preprocessor,
	classifier *nlp.Classifier,
	extractor *nlp.EntityExtractor,
	m *metrics.AgentMetrics,
	logger *slog.Logger,
) *Agent {
	memory := NewMemory(cfg.MaxMemoryMessages, preprocessor)
	memory.AddSystem(buildSystemPrompt(toolReg))
	return &Agent{
		id:         uuid.NewString(),
		cfg:        cfg,
		llmClient:  llmClient,
		toolReg:    toolReg,
		memory:     memory,
		nlpProc:    preprocessor,
		classifier: classifier,
		extractor:  extractor,
		metrics:    m,
		limiter:    newRateLimiter(cfg.RateLimitRPS, cfg.RateLimitBurst),
		logger:     logger,
		workerSem:  make(chan struct{}, cfg.MaxWorkers),
	}
}

func (a *Agent) ID() string { return a.id }

func (a *Agent) Run(ctx context.Context, task *Task) (*TaskResult, error) {
	start := time.Now()
	result := &TaskResult{TaskID: task.ID, Status: StatusRunning}

	if err := a.limiter.Wait(ctx); err != nil {
		result.Status = StatusCancelled
		result.Error = "rate limit cancelled: " + err.Error()
		return result, nil
	}

	taskCtx, cancel := context.WithTimeout(ctx, a.cfg.TaskTimeout)
	defer cancel()

	a.metrics.TasksStarted.Inc()
	a.logger.Info("task started", "task_id", task.ID, "prompt", truncate(task.Prompt, 80))

	processed := a.nlpProc.Process(task.Prompt)
	a.logger.Debug("NLP processed", "words", processed.WordCount, "lang", processed.Language)

	a.memory.Add("user", task.Prompt, map[string]any{"task_id": task.ID})

	const maxIter = 5
	for iter := 0; iter < maxIter; iter++ {
		if taskCtx.Err() != nil {
			break
		}
		resp, err := a.llmClient.Complete(taskCtx, &llmclient.CompleteRequest{
			Prompt:    task.Prompt,
			History:   a.memory.Messages(),
			SessionID: task.SessionID,
		})
		if err != nil {
			result.Status = StatusFailed
			result.Error = fmt.Sprintf("LLM completion failed: %v", err)
			a.metrics.TasksFailed.Inc()
			return result, nil
		}
		result.TokensUsed += resp.TokensUsed

		toolCalls := parseToolCalls(resp.Text)
		if len(toolCalls) == 0 {
			a.memory.Add("assistant", resp.Text, nil)
			result.Status = StatusCompleted
			result.Response = cleanFinalAnswer(resp.Text)
			break
		}

		toolResults := a.executeToolsParallel(taskCtx, toolCalls)
		result.ToolCalls = append(result.ToolCalls, toolResults...)

		a.memory.Add("assistant", resp.Text, nil)
		a.memory.Add("user", "[Tool results]\n"+buildToolResultsSummary(toolResults), nil)
	}

	if result.Status == StatusRunning {
		result.Status = StatusCompleted
		result.Response = "Task processing completed."
	}

	result.Duration = time.Since(start)
	result.CompletedAt = time.Now()
	a.metrics.TasksCompleted.Inc()
	a.metrics.TaskDuration.Observe(result.Duration.Seconds())
	a.metrics.TokensUsed.Add(float64(result.TokensUsed))

	a.logger.Info("task completed",
		"task_id", task.ID,
		"duration_ms", result.Duration.Milliseconds(),
		"tool_calls", len(result.ToolCalls),
		"tokens", result.TokensUsed,
	)
	return result, nil
}

func (a *Agent) executeToolsParallel(ctx context.Context, calls []tools.ToolInput) []ToolCallRecord {
	var wg sync.WaitGroup
	var mu sync.Mutex
	records := make([]ToolCallRecord, 0, len(calls))

	for _, call := range calls {
		wg.Add(1)
		go func(input tools.ToolInput) {
			defer wg.Done()
			select {
			case a.workerSem <- struct{}{}:
				defer func() { <-a.workerSem }()
			case <-ctx.Done():
				return
			}
			rec := a.executeSingleTool(ctx, input)
			mu.Lock()
			records = append(records, rec)
			mu.Unlock()
		}(call)
	}
	wg.Wait()
	return records
}

func (a *Agent) executeSingleTool(ctx context.Context, input tools.ToolInput) ToolCallRecord {
	start := time.Now()
	rec := ToolCallRecord{ToolName: input.Name, Params: input.Params}

	tool, ok := a.toolReg.Get(input.Name)
	if !ok {
		rec.Output = fmt.Sprintf("unknown tool: %s", input.Name)
		rec.IsError = true
		a.metrics.ToolCallErrors.Inc()
		return rec
	}
	a.metrics.ToolCalls.Inc()

	output, err := tool.Execute(ctx, input)
	rec.Duration = time.Since(start)
	if err != nil {
		rec.Output = err.Error()
		rec.IsError = true
		a.metrics.ToolCallErrors.Inc()
		a.logger.Warn("tool error", "tool", input.Name, "error", err)
		return rec
	}
	rec.Output = output.String()
	rec.IsError = output.IsError
	a.logger.Debug("tool executed", "tool", input.Name, "duration_ms", rec.Duration.Milliseconds())
	return rec
}

func (a *Agent) Memory() *Memory { return a.memory }

func buildSystemPrompt(reg *tools.Registry) string {
	var sb strings.Builder
	sb.WriteString("You are a helpful AI agent. You have access to the following tools:\n\n")
	for _, t := range reg.List() {
		sb.WriteString(fmt.Sprintf("- **%s**: %s\n", t.Name(), t.Description()))
	}
	sb.WriteString(`
To use a tool, output a JSON block like this:
<tool_call>
{"name": "tool_name", "params": {"key": "value"}}
</tool_call>

You may call multiple tools in parallel. Once done, give your final answer without tool_call blocks.
`)
	return sb.String()
}

func parseToolCalls(text string) []tools.ToolInput {
	var calls []tools.ToolInput
	const open, close = "<tool_call>", "</tool_call>"
	remaining := text
	for {
		start := strings.Index(remaining, open)
		if start < 0 {
			break
		}
		end := strings.Index(remaining, close)
		if end < 0 {
			break
		}
		raw := strings.TrimSpace(remaining[start+len(open) : end])
		var call struct {
			Name   string            `json:"name"`
			Params map[string]string `json:"params"`
		}
		if err := json.Unmarshal([]byte(raw), &call); err == nil && call.Name != "" {
			calls = append(calls, tools.ToolInput{Name: call.Name, Params: call.Params})
		}
		remaining = remaining[end+len(close):]
	}
	return calls
}

func buildToolResultsSummary(records []ToolCallRecord) string {
	var sb strings.Builder
	for _, r := range records {
		status := "ok"
		if r.IsError {
			status = "error"
		}
		sb.WriteString(fmt.Sprintf("Tool: %s [%s]\nOutput: %s\n\n", r.ToolName, status, truncate(r.Output, 500)))
	}
	return sb.String()
}

func cleanFinalAnswer(text string) string {
	re := strings.NewReplacer("<tool_call>", "", "</tool_call>", "")
	return strings.TrimSpace(re.Replace(text))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
