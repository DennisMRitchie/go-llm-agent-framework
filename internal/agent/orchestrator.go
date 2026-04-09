package agent

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/DennisMRitchie/go-llm-agent-framework/internal/config"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/llmclient"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/metrics"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/nlp"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/tools"
)

type taskEntry struct {
	task   *Task
	result *TaskResult
	status TaskStatus
	cancel context.CancelFunc
	done   chan struct{}
}

type Orchestrator struct {
	cfg          config.AgentConfig
	llmClient    llmclient.Client
	toolReg      *tools.Registry
	preprocessor *nlp.Preprocessor
	classifier   *nlp.Classifier
	extractor    *nlp.EntityExtractor
	m            *metrics.AgentMetrics
	logger       *slog.Logger
	mu           sync.RWMutex
	tasks        map[string]*taskEntry
	runSem       chan struct{}
}

func NewOrchestrator(
	cfg config.AgentConfig,
	llmClient llmclient.Client,
	toolReg *tools.Registry,
	preprocessor *nlp.Preprocessor,
	classifier *nlp.Classifier,
	extractor *nlp.EntityExtractor,
	m *metrics.AgentMetrics,
	logger *slog.Logger,
) *Orchestrator {
	return &Orchestrator{
		cfg:          cfg,
		llmClient:    llmClient,
		toolReg:      toolReg,
		preprocessor: preprocessor,
		classifier:   classifier,
		extractor:    extractor,
		m:            m,
		logger:       logger,
		tasks:        make(map[string]*taskEntry),
		runSem:       make(chan struct{}, cfg.MaxConcurrentRuns),
	}
}

func (o *Orchestrator) Submit(ctx context.Context, prompt, sessionID string, context_ map[string]string) (string, error) {
	taskID := uuid.NewString()
	task := &Task{ID: taskID, Prompt: prompt, SessionID: sessionID, Context: context_, CreatedAt: time.Now()}

	taskCtx, cancel := context.WithCancel(ctx)
	entry := &taskEntry{task: task, status: StatusPending, cancel: cancel, done: make(chan struct{})}

	o.mu.Lock()
	o.tasks[taskID] = entry
	o.mu.Unlock()

	o.m.QueueDepth.Inc()
	go o.runTask(taskCtx, entry)
	o.logger.Info("task submitted", "task_id", taskID)
	return taskID, nil
}

func (o *Orchestrator) SubmitAndWait(ctx context.Context, prompt, sessionID string, context_ map[string]string) (*TaskResult, error) {
	taskID, err := o.Submit(ctx, prompt, sessionID, context_)
	if err != nil {
		return nil, err
	}
	return o.Wait(ctx, taskID)
}

func (o *Orchestrator) Wait(ctx context.Context, taskID string) (*TaskResult, error) {
	o.mu.RLock()
	entry, ok := o.tasks[taskID]
	o.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("task %s not found", taskID)
	}
	select {
	case <-entry.done:
		return entry.result, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (o *Orchestrator) Status(taskID string) (TaskStatus, *TaskResult, error) {
	o.mu.RLock()
	defer o.mu.RUnlock()
	entry, ok := o.tasks[taskID]
	if !ok {
		return "", nil, fmt.Errorf("task %s not found", taskID)
	}
	return entry.status, entry.result, nil
}

func (o *Orchestrator) Cancel(taskID string) error {
	o.mu.RLock()
	entry, ok := o.tasks[taskID]
	o.mu.RUnlock()
	if !ok {
		return fmt.Errorf("task %s not found", taskID)
	}
	entry.cancel()
	return nil
}

func (o *Orchestrator) ListTasks() []*TaskResult {
	o.mu.RLock()
	defer o.mu.RUnlock()
	results := make([]*TaskResult, 0, len(o.tasks))
	for _, e := range o.tasks {
		if e.result != nil {
			results = append(results, e.result)
		} else {
			results = append(results, &TaskResult{TaskID: e.task.ID, Status: e.status})
		}
	}
	return results
}

func (o *Orchestrator) Cleanup(maxAge time.Duration) int {
	cutoff := time.Now().Add(-maxAge)
	o.mu.Lock()
	defer o.mu.Unlock()
	removed := 0
	for id, e := range o.tasks {
		if (e.status == StatusCompleted || e.status == StatusFailed || e.status == StatusCancelled) &&
			e.result != nil && e.result.CompletedAt.Before(cutoff) {
			delete(o.tasks, id)
			removed++
		}
	}
	return removed
}

func (o *Orchestrator) runTask(ctx context.Context, entry *taskEntry) {
	select {
	case o.runSem <- struct{}{}:
		defer func() { <-o.runSem }()
	case <-ctx.Done():
		o.mu.Lock()
		entry.status = StatusCancelled
		entry.result = &TaskResult{TaskID: entry.task.ID, Status: StatusCancelled, Error: "cancelled before start"}
		o.mu.Unlock()
		close(entry.done)
		o.m.QueueDepth.Dec()
		return
	}

	o.mu.Lock()
	entry.status = StatusRunning
	o.mu.Unlock()
	o.m.QueueDepth.Dec()

	a := New(o.cfg, o.llmClient, o.toolReg, o.preprocessor, o.classifier, o.extractor, o.m, o.logger)
	result, err := a.Run(ctx, entry.task)
	if err != nil {
		result = &TaskResult{TaskID: entry.task.ID, Status: StatusFailed, Error: err.Error()}
	}

	o.mu.Lock()
	entry.result = result
	entry.status = result.Status
	o.mu.Unlock()
	close(entry.done)

	o.logger.Info("task finished", "task_id", entry.task.ID, "status", result.Status, "duration_ms", result.Duration.Milliseconds())
}
