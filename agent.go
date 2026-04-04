package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
	"golang.org/x/time/rate"

	"github.com/DennisMRitchie/go-llm-agent-framework/internal/config"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/llmclient"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/metrics"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/nlp"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/tools"
)

var tracer = otel.Tracer("agent")

// --- Task types ---

// TaskStatus represents the lifecycle of an agent task.
type TaskStatus string

const (
	StatusPending   TaskStatus = "pending"
	StatusRunning   TaskStatus = "running"
	StatusCompleted TaskStatus = "completed"
	StatusFailed    TaskStatus = "failed"
	StatusCancelled TaskStatus = "cancelled"
)

// Task is a unit of work submitted to an agent.
type Task struct {
	ID        string
	Prompt    string
	SessionID string
	Context   map[string]string
	CreatedAt time.Time
}

// TaskResult holds the outcome of a task execution.
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

// ToolCallRecord logs a single tool invocation.
type ToolCallRecord struct {
	ToolName  string
	Params    map[string]string
	Output    string
	IsError   bool
	Duration  time.Duration
}

// --- Agent ---

// Agent is a stateful LLM agent with tool-calling, memory, and NLP capabilities.
type Agent struct {
	id          string
	cfg         config.AgentConfig
	llmClient   llmclient.Client
	toolReg     *tools.Registry
	memory      *Memory
	nlp         *nlp.Preprocessor
	classifier  *nlp.Classifier
	extractor   *nlp.EntityExtractor
	metrics     *metrics.AgentMetrics
	rateLimiter *rate.Limiter
	logger      *zap.Logger

	// Worker pool semaphore for parallel tool execution
	workerSem chan struct{}
}

// New creates a new Agent.
func New(
	cfg config.AgentConfig,
	llmClient llmclient.Client,
	toolReg *tools.Registry,
	preprocessor *nlp.Preprocessor,
	classifier *nlp.Classifier,
	extractor *nlp.EntityExtractor,
	m *metrics.AgentMetrics,
	logger *zap.Logger,
) *Agent {
	memory := NewMemory(cfg.MaxMemoryMessages, preprocessor)
	memory.AddSystem(buildSystemPrompt(toolReg))

	return &Agent{
		id:          uuid.NewString(),
		cfg:         cfg,
		llmClient:   llmClient,
		toolReg:     toolReg,
		memory:      memory,
		nlp:         preprocessor,
		classifier:  classifier,
		extractor:   extractor,
		metrics:     m,
		rateLimiter: rate.NewLimiter(rate.Limit(cfg.RateLimitRPS), cfg.RateLimitBurst),
		logger:      logger.With(zap.String("agent_id", "")),
		workerSem:   make(chan struct{}, cfg.MaxWorkers),
	}
}

// ID returns the agent's unique identifier.
func (a *Agent) ID() string { return a.id }

// Run executes a task through the ReAct (Reason + Act) loop.
func (a *Agent) Run(ctx context.Context, task *Task) (*TaskResult, error) {
	ctx, span := tracer.Start(ctx, "agent.Run",
		trace.WithAttributes(
			attribute.String("task_id", task.ID),
			attribute.String("session_id", task.SessionID),
		))
	defer span.End()

	start := time.Now()
	result := &TaskResult{
		TaskID: task.ID,
		Status: StatusRunning,
	}

	// Rate limiting
	if err := a.rateLimiter.Wait(ctx); err != nil {
		result.Status = StatusCancelled
		result.Error = "rate limit cancelled: " + err.Error()
		return result, nil
	}

	// Task timeout
	taskCtx, cancel := context.WithTimeout(ctx, a.cfg.TaskTimeout)
	defer cancel()

	a.metrics.TasksStarted.Inc()
	a.logger.Info("agent task started", zap.String("task_id", task.ID), zap.String("prompt", truncate(task.Prompt, 80)))

	// NLP preprocessing
	processed := a.nlp.Process(task.Prompt)
	a.logger.Debug("NLP processed",
		zap.Int("word_count", processed.WordCount),
		zap.String("language", processed.Language),
	)

	// Add user message to memory
	a.memory.Add("user", task.Prompt, map[string]any{
		"task_id": task.ID,
		"context": task.Context,
	})

	// ReAct loop — max 5 iterations to prevent runaway
	const maxIter = 5
	for iter := 0; iter < maxIter; iter++ {
		if taskCtx.Err() != nil {
			break
		}

		// Get LLM completion
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

		// Parse tool calls from response
		toolCalls := parseToolCalls(resp.Text)
		if len(toolCalls) == 0 {
			// No tool calls — final answer
			a.memory.Add("assistant", resp.Text, nil)
			result.Status = StatusCompleted
			result.Response = cleanFinalAnswer(resp.Text)
			break
		}

		// Execute tool calls in parallel
		toolResults := a.executeToolsParallel(taskCtx, toolCalls)
		result.ToolCalls = append(result.ToolCalls, toolResults...)

		// Build tool results summary for next LLM iteration
		summary := buildToolResultsSummary(toolResults)
		a.memory.Add("assistant", resp.Text, nil)
		a.memory.Add("user", "[Tool results]\n"+summary, nil)
	}

	if result.Status == StatusRunning {
		// Loop exhausted without terminal answer
		result.Status = StatusCompleted
		result.Response = "Task processing completed (max iterations reached)."
	}

	result.Duration = time.Since(start)
	result.CompletedAt = time.Now()

	a.metrics.TasksCompleted.Inc()
	a.metrics.TaskDuration.Observe(result.Duration.Seconds())
	a.metrics.TokensUsed.Add(float64(result.TokensUsed))

	a.logger.Info("agent task completed",
		zap.String("task_id", task.ID),
		zap.Duration("duration", result.Duration),
		zap.Int("tool_calls", len(result.ToolCalls)),
		zap.Int("tokens", result.TokensUsed),
	)

	span.SetAttributes(
		attribute.Int("tokens_used", result.TokensUsed),
		attribute.Int("tool_calls", len(result.ToolCalls)),
		attribute.String("status", string(result.Status)),
	)

	return result, nil
}

// executeToolsParallel runs multiple tool calls concurrently using the worker pool.
func (a *Agent) executeToolsParallel(ctx context.Context, calls []tools.ToolInput) []ToolCallRecord {
	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		records = make([]ToolCallRecord, 0, len(calls))
	)

	for _, call := range calls {
		wg.Add(1)
		go func(input tools.ToolInput) {
			defer wg.Done()

			// Acquire worker slot
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
	rec := ToolCallRecord{
		ToolName: input.Name,
		Params:   input.Params,
	}

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
		a.logger.Warn("tool execution error", zap.String("tool", input.Name), zap.Error(err))
		return rec
	}

	rec.Output = output.String()
	rec.IsError = output.IsError

	a.logger.Debug("tool executed",
		zap.String("tool", input.Name),
		zap.Duration("duration", rec.Duration),
		zap.Bool("error", rec.IsError),
	)
	return rec
}

// Memory returns the agent's memory (read-only access for inspection).
func (a *Agent) Memory() *Memory { return a.memory }

// --- helpers ---

// buildSystemPrompt creates the system prompt with available tools.
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

You may call multiple tools in parallel by including multiple tool_call blocks.
Once you have enough information, provide your final answer without any tool_call blocks.
`)
	return sb.String()
}

// parseToolCalls extracts all <tool_call>...</tool_call> blocks from LLM output.
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
	// Strip any leftover tool_call tags
	re := strings.NewReplacer("<tool_call>", "", "</tool_call>", "")
	return strings.TrimSpace(re.Replace(text))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
