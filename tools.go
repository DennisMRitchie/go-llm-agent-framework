// Package tools defines the Tool interface, Registry, and built-in tool implementations.
package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"
)

// --- Core types ---

// ToolInput carries the raw parameters for a tool invocation.
type ToolInput struct {
	Name   string            `json:"name"`
	Params map[string]string `json:"params"`
}

// ToolOutput is what a tool returns.
type ToolOutput struct {
	Content   string        `json:"content"`
	IsError   bool          `json:"is_error,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
	Duration  time.Duration `json:"duration_ms,omitempty"`
}

func (o *ToolOutput) String() string {
	if o.IsError {
		return "[error] " + o.Content
	}
	return o.Content
}

// Tool is the interface all tools must satisfy.
type Tool interface {
	// Name returns the unique tool identifier.
	Name() string
	// Description explains what the tool does (shown to the LLM).
	Description() string
	// Schema returns the JSON schema for parameters.
	Schema() map[string]any
	// Execute runs the tool with the given input.
	Execute(ctx context.Context, input ToolInput) (*ToolOutput, error)
}

// --- Registry ---

// Registry is a thread-safe store of named tools.
type Registry struct {
	mu    sync.RWMutex
	tools map[string]Tool
}

func NewRegistry() *Registry {
	r := &Registry{tools: make(map[string]Tool)}
	// Register built-ins
	r.Register(&CalculatorTool{})
	r.Register(&TextSummarizerTool{})
	r.Register(&DateTimeTool{})
	r.Register(&JSONParseTool{})
	r.Register(&EchoTool{})
	return r
}

func (r *Registry) Register(t Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools[t.Name()] = t
}

func (r *Registry) Get(name string) (Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.tools[name]
	return t, ok
}

func (r *Registry) List() []Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]Tool, 0, len(r.tools))
	for _, t := range r.tools {
		out = append(out, t)
	}
	return out
}

// ToolSchemas returns all tool schemas in OpenAI function-calling format.
func (r *Registry) ToolSchemas() []map[string]any {
	tools := r.List()
	out := make([]map[string]any, len(tools))
	for i, t := range tools {
		out[i] = map[string]any{
			"name":        t.Name(),
			"description": t.Description(),
			"parameters":  t.Schema(),
		}
	}
	return out
}

// --- Built-in: Calculator ---

type CalculatorTool struct{}

func (c *CalculatorTool) Name()        string { return "calculator" }
func (c *CalculatorTool) Description() string {
	return "Performs basic arithmetic: add, subtract, multiply, divide, power, sqrt."
}
func (c *CalculatorTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"operation": map[string]any{"type": "string", "enum": []string{"add", "subtract", "multiply", "divide", "power", "sqrt"}},
			"a":         map[string]any{"type": "number"},
			"b":         map[string]any{"type": "number", "description": "Not required for sqrt"},
		},
		"required": []string{"operation", "a"},
	}
}

func (c *CalculatorTool) Execute(_ context.Context, input ToolInput) (*ToolOutput, error) {
	start := time.Now()
	a, err := strconv.ParseFloat(input.Params["a"], 64)
	if err != nil {
		return &ToolOutput{Content: "invalid 'a' parameter", IsError: true}, nil
	}

	var result float64
	op := strings.ToLower(input.Params["operation"])
	switch op {
	case "sqrt":
		if a < 0 {
			return &ToolOutput{Content: "cannot sqrt negative number", IsError: true}, nil
		}
		result = math.Sqrt(a)
	default:
		b, err := strconv.ParseFloat(input.Params["b"], 64)
		if err != nil {
			return &ToolOutput{Content: "invalid 'b' parameter", IsError: true}, nil
		}
		switch op {
		case "add":
			result = a + b
		case "subtract":
			result = a - b
		case "multiply":
			result = a * b
		case "divide":
			if b == 0 {
				return &ToolOutput{Content: "division by zero", IsError: true}, nil
			}
			result = a / b
		case "power":
			result = math.Pow(a, b)
		default:
			return &ToolOutput{Content: fmt.Sprintf("unknown operation: %s", op), IsError: true}, nil
		}
	}

	return &ToolOutput{
		Content:  strconv.FormatFloat(result, 'f', -1, 64),
		Duration: time.Since(start),
		Metadata: map[string]any{"operation": op, "result": result},
	}, nil
}

// --- Built-in: TextSummarizer (rule-based, no LLM call) ---

type TextSummarizerTool struct{}

func (t *TextSummarizerTool) Name()        string { return "text_summarizer" }
func (t *TextSummarizerTool) Description() string {
	return "Extracts the first N sentences from text as a simple summary."
}
func (t *TextSummarizerTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"text":     map[string]any{"type": "string"},
			"max_sents": map[string]any{"type": "integer", "default": 3},
		},
		"required": []string{"text"},
	}
}

func (t *TextSummarizerTool) Execute(_ context.Context, input ToolInput) (*ToolOutput, error) {
	start := time.Now()
	text := input.Params["text"]
	if text == "" {
		return &ToolOutput{Content: "empty text", IsError: true}, nil
	}

	maxSents := 3
	if n, err := strconv.Atoi(input.Params["max_sents"]); err == nil && n > 0 {
		maxSents = n
	}

	sentences := splitSentences(text)
	if len(sentences) > maxSents {
		sentences = sentences[:maxSents]
	}
	summary := strings.Join(sentences, " ")

	return &ToolOutput{
		Content:  summary,
		Duration: time.Since(start),
		Metadata: map[string]any{"sentence_count": len(sentences), "char_count": len(text)},
	}, nil
}

func splitSentences(text string) []string {
	var sentences []string
	var buf strings.Builder
	for i, ch := range text {
		buf.WriteRune(ch)
		if ch == '.' || ch == '!' || ch == '?' {
			next := i + 1
			if next < len(text) && (text[next] == ' ' || text[next] == '\n') {
				s := strings.TrimSpace(buf.String())
				if s != "" {
					sentences = append(sentences, s)
				}
				buf.Reset()
			}
		}
	}
	if rem := strings.TrimSpace(buf.String()); rem != "" {
		sentences = append(sentences, rem)
	}
	return sentences
}

// --- Built-in: DateTime ---

type DateTimeTool struct{}

func (d *DateTimeTool) Name()        string { return "datetime" }
func (d *DateTimeTool) Description() string {
	return "Returns the current date and time in the requested timezone and format."
}
func (d *DateTimeTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"timezone": map[string]any{"type": "string", "default": "UTC"},
			"format":   map[string]any{"type": "string", "default": "RFC3339"},
		},
	}
}

func (d *DateTimeTool) Execute(_ context.Context, input ToolInput) (*ToolOutput, error) {
	tz := input.Params["timezone"]
	if tz == "" {
		tz = "UTC"
	}
	loc, err := time.LoadLocation(tz)
	if err != nil {
		return &ToolOutput{Content: fmt.Sprintf("unknown timezone: %s", tz), IsError: true}, nil
	}
	now := time.Now().In(loc)

	format := input.Params["format"]
	var result string
	switch strings.ToUpper(format) {
	case "RFC3339", "":
		result = now.Format(time.RFC3339)
	case "DATE":
		result = now.Format("2006-01-02")
	case "TIME":
		result = now.Format("15:04:05")
	case "UNIX":
		result = strconv.FormatInt(now.Unix(), 10)
	default:
		result = now.Format(format)
	}

	return &ToolOutput{
		Content: result,
		Metadata: map[string]any{"timezone": tz, "unix": now.Unix()},
	}, nil
}

// --- Built-in: JSONParse ---

type JSONParseTool struct{}

func (j *JSONParseTool) Name()        string { return "json_parse" }
func (j *JSONParseTool) Description() string {
	return "Parses a JSON string and returns a field value by dot-notation path."
}
func (j *JSONParseTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"json": map[string]any{"type": "string"},
			"path": map[string]any{"type": "string", "description": "Dot-separated path, e.g. 'user.name'"},
		},
		"required": []string{"json"},
	}
}

func (j *JSONParseTool) Execute(_ context.Context, input ToolInput) (*ToolOutput, error) {
	raw := input.Params["json"]
	if raw == "" {
		return &ToolOutput{Content: "empty JSON", IsError: true}, nil
	}

	var obj any
	if err := json.Unmarshal([]byte(raw), &obj); err != nil {
		return &ToolOutput{Content: "invalid JSON: " + err.Error(), IsError: true}, nil
	}

	path := input.Params["path"]
	if path == "" {
		pretty, _ := json.MarshalIndent(obj, "", "  ")
		return &ToolOutput{Content: string(pretty)}, nil
	}

	// Traverse dot path
	parts := strings.Split(path, ".")
	current := obj
	for _, part := range parts {
		m, ok := current.(map[string]any)
		if !ok {
			return &ToolOutput{Content: fmt.Sprintf("path '%s' not found", path), IsError: true}, nil
		}
		current, ok = m[part]
		if !ok {
			return &ToolOutput{Content: fmt.Sprintf("key '%s' not found", part), IsError: true}, nil
		}
	}

	result, _ := json.Marshal(current)
	return &ToolOutput{Content: string(result)}, nil
}

// --- Built-in: Echo (useful for testing) ---

type EchoTool struct{}

func (e *EchoTool) Name()        string { return "echo" }
func (e *EchoTool) Description() string { return "Returns the input message unchanged. Useful for testing." }
func (e *EchoTool) Schema() map[string]any {
	return map[string]any{
		"type":       "object",
		"properties": map[string]any{"message": map[string]any{"type": "string"}},
		"required":   []string{"message"},
	}
}

func (e *EchoTool) Execute(_ context.Context, input ToolInput) (*ToolOutput, error) {
	return &ToolOutput{Content: input.Params["message"]}, nil
}
