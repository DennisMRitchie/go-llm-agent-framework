package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// AgentMetrics holds metrics for the agent
type AgentMetrics struct {
	TasksStarted   prometheus.Counter
	TasksCompleted prometheus.Counter
	TasksFailed    prometheus.Counter
	TaskDuration   prometheus.Histogram
	TokensUsed     prometheus.Counter
	ToolCallErrors prometheus.Counter
	ToolCalls      prometheus.Counter
	QueueDepth     prometheus.Gauge
}

// NewAgentMetrics creates new agent metrics
func NewAgentMetrics() *AgentMetrics {
	return &AgentMetrics{
		TasksStarted: promauto.NewCounter(prometheus.CounterOpts{
			Name: "agent_tasks_started_total",
			Help: "Total number of tasks started",
		}),
		TasksCompleted: promauto.NewCounter(prometheus.CounterOpts{
			Name: "agent_tasks_completed_total",
			Help: "Total number of tasks completed",
		}),
		TasksFailed: promauto.NewCounter(prometheus.CounterOpts{
			Name: "agent_tasks_failed_total",
			Help: "Total number of tasks failed",
		}),
		TaskDuration: promauto.NewHistogram(prometheus.HistogramOpts{
			Name: "agent_task_duration_seconds",
			Help: "Duration of agent tasks",
		}),
		TokensUsed: promauto.NewCounter(prometheus.CounterOpts{
			Name: "agent_tokens_used_total",
			Help: "Total number of tokens used",
		}),
		ToolCallErrors: promauto.NewCounter(prometheus.CounterOpts{
			Name: "agent_tool_call_errors_total",
			Help: "Total number of tool call errors",
		}),
		ToolCalls: promauto.NewCounter(prometheus.CounterOpts{
			Name: "agent_tool_calls_total",
			Help: "Total number of tool calls",
		}),
		QueueDepth: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "agent_queue_depth",
			Help: "Current depth of the agent task queue",
		}),
	}
}

// HTTPMetrics holds metrics for HTTP requests
type HTTPMetrics struct {
	RequestsTotal   *prometheus.CounterVec
	RequestDuration *prometheus.HistogramVec
	ActiveRequests  prometheus.Gauge
}

// NewHTTPMetrics creates new HTTP metrics
func NewHTTPMetrics() *HTTPMetrics {
	return &HTTPMetrics{
		RequestsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Total number of HTTP requests",
		}, []string{"method", "endpoint", "status"}),
		RequestDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Name: "http_request_duration_seconds",
			Help: "Duration of HTTP requests",
		}, []string{"method", "endpoint"}),
		ActiveRequests: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "http_active_requests",
			Help: "Number of active HTTP requests",
		}),
	}
}
