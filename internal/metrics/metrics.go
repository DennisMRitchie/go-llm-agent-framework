// Package metrics provides lightweight Prometheus-compatible metrics
// using only sync/atomic — no external dependencies.
package metrics

import (
	"fmt"
	"io"
	"math"
	"sync"
	"sync/atomic"
)

// Counter is a monotonically increasing float64 counter.
type Counter struct{ bits atomic.Uint64 }

func (c *Counter) Inc() { c.Add(1) }
func (c *Counter) Add(v float64) {
	for {
		old := c.bits.Load()
		newVal := math.Float64bits(math.Float64frombits(old) + v)
		if c.bits.CompareAndSwap(old, newVal) {
			return
		}
	}
}
func (c *Counter) Value() float64 { return math.Float64frombits(c.bits.Load()) }

// Gauge is a float64 that can go up or down.
type Gauge struct{ bits atomic.Uint64 }

func (g *Gauge) Set(v float64) { g.bits.Store(math.Float64bits(v)) }
func (g *Gauge) Inc()          { g.Add(1) }
func (g *Gauge) Dec()          { g.Add(-1) }
func (g *Gauge) Add(v float64) {
	for {
		old := g.bits.Load()
		newVal := math.Float64bits(math.Float64frombits(old) + v)
		if g.bits.CompareAndSwap(old, newVal) {
			return
		}
	}
}
func (g *Gauge) Value() float64 { return math.Float64frombits(g.bits.Load()) }

// Histogram tracks observations in fixed buckets.
type Histogram struct {
	mu      sync.Mutex
	buckets []float64 // upper bounds
	counts  []uint64
	sum     float64
	total   uint64
}

var defaultBuckets = []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10}

func NewHistogram(buckets ...float64) *Histogram {
	if len(buckets) == 0 {
		buckets = defaultBuckets
	}
	return &Histogram{buckets: buckets, counts: make([]uint64, len(buckets)+1)}
}

func (h *Histogram) Observe(v float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.sum += v
	h.total++
	for i, b := range h.buckets {
		if v <= b {
			h.counts[i]++
			return
		}
	}
	h.counts[len(h.buckets)]++ // +Inf bucket
}

// CounterVec is a labeled set of counters.
type CounterVec struct {
	mu      sync.RWMutex
	labels  []string
	entries map[string]*Counter
}

func NewCounterVec(labels ...string) *CounterVec {
	return &CounterVec{labels: labels, entries: make(map[string]*Counter)}
}

func (cv *CounterVec) With(labelValues ...string) *Counter {
	key := fmt.Sprint(labelValues)
	cv.mu.RLock()
	c, ok := cv.entries[key]
	cv.mu.RUnlock()
	if ok {
		return c
	}
	cv.mu.Lock()
	defer cv.mu.Unlock()
	if c, ok = cv.entries[key]; !ok {
		c = &Counter{}
		cv.entries[key] = c
	}
	return c
}

// HistogramVec is a labeled set of histograms.
type HistogramVec struct {
	mu      sync.RWMutex
	labels  []string
	entries map[string]*Histogram
}

func NewHistogramVec(labels ...string) *HistogramVec {
	return &HistogramVec{labels: labels, entries: make(map[string]*Histogram)}
}

func (hv *HistogramVec) With(labelValues ...string) *Histogram {
	key := fmt.Sprint(labelValues)
	hv.mu.RLock()
	h, ok := hv.entries[key]
	hv.mu.RUnlock()
	if ok {
		return h
	}
	hv.mu.Lock()
	defer hv.mu.Unlock()
	if h, ok = hv.entries[key]; !ok {
		h = NewHistogram()
		hv.entries[key] = h
	}
	return h
}

// --- Agent metrics ---

type AgentMetrics struct {
	TasksStarted   *Counter
	TasksCompleted *Counter
	TasksFailed    *Counter
	TaskDuration   *Histogram
	TokensUsed     *Counter
	ToolCalls      *Counter
	ToolCallErrors *Counter
	QueueDepth     *Gauge
}

func NewAgentMetrics() *AgentMetrics {
	return &AgentMetrics{
		TasksStarted:   &Counter{},
		TasksCompleted: &Counter{},
		TasksFailed:    &Counter{},
		TaskDuration:   NewHistogram(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
		TokensUsed:     &Counter{},
		ToolCalls:      &Counter{},
		ToolCallErrors: &Counter{},
		QueueDepth:     &Gauge{},
	}
}

// --- HTTP metrics ---

type HTTPMetrics struct {
	RequestsTotal   *CounterVec
	RequestDuration *HistogramVec
	ActiveRequests  *Gauge
}

func NewHTTPMetrics() *HTTPMetrics {
	return &HTTPMetrics{
		RequestsTotal:   NewCounterVec("method", "path", "status"),
		RequestDuration: NewHistogramVec("method", "path"),
		ActiveRequests:  &Gauge{},
	}
}

// --- Prometheus text format writer ---

type Registry struct {
	mu      sync.RWMutex
	metrics []namedMetric
}

type namedMetric struct {
	name   string
	help   string
	mtype  string
	metric any
}

var DefaultRegistry = &Registry{}

func Register(name, help, mtype string, m any) {
	DefaultRegistry.mu.Lock()
	defer DefaultRegistry.mu.Unlock()
	DefaultRegistry.metrics = append(DefaultRegistry.metrics, namedMetric{name, help, mtype, m})
}

func WriteMetrics(w io.Writer) {
	DefaultRegistry.mu.RLock()
	defer DefaultRegistry.mu.RUnlock()
	for _, nm := range DefaultRegistry.metrics {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s %s\n", nm.name, nm.help, nm.name, nm.mtype)
		switch m := nm.metric.(type) {
		case *Counter:
			fmt.Fprintf(w, "%s %.6g\n", nm.name, m.Value())
		case *Gauge:
			fmt.Fprintf(w, "%s %.6g\n", nm.name, m.Value())
		case *Histogram:
			m.mu.Lock()
			cumulative := uint64(0)
			for i, b := range m.buckets {
				cumulative += m.counts[i]
				fmt.Fprintf(w, "%s_bucket{le=\"%g\"} %d\n", nm.name, b, cumulative)
			}
			cumulative += m.counts[len(m.buckets)]
			fmt.Fprintf(w, "%s_bucket{le=\"+Inf\"} %d\n", nm.name, cumulative)
			fmt.Fprintf(w, "%s_sum %.6g\n%s_count %d\n", nm.name, m.sum, nm.name, m.total)
			m.mu.Unlock()
		case *CounterVec:
			m.mu.RLock()
			for k, c := range m.entries {
				fmt.Fprintf(w, "%s%s %.6g\n", nm.name, k, c.Value())
			}
			m.mu.RUnlock()
		}
	}
}

func RegisterAgentMetrics(m *AgentMetrics) {
	Register("llmagent_tasks_started_total", "Tasks started", "counter", m.TasksStarted)
	Register("llmagent_tasks_completed_total", "Tasks completed", "counter", m.TasksCompleted)
	Register("llmagent_tasks_failed_total", "Tasks failed", "counter", m.TasksFailed)
	Register("llmagent_task_duration_seconds", "Task duration", "histogram", m.TaskDuration)
	Register("llmagent_tokens_used_total", "Tokens used", "counter", m.TokensUsed)
	Register("llmagent_tool_calls_total", "Tool calls", "counter", m.ToolCalls)
	Register("llmagent_tool_call_errors_total", "Tool errors", "counter", m.ToolCallErrors)
	Register("llmagent_queue_depth", "Queue depth", "gauge", m.QueueDepth)
}

func RegisterHTTPMetrics(m *HTTPMetrics) {
	Register("llmagent_http_requests_total", "HTTP requests", "counter", m.RequestsTotal)
	Register("llmagent_http_request_duration_seconds", "HTTP latency", "histogram", m.RequestDuration)
	Register("llmagent_http_active_requests", "Active requests", "gauge", m.ActiveRequests)
}
