package tests

import (
	"context"
	"strings"
	"testing"

	"github.com/DennisMRitchie/go-llm-agent-framework/internal/config"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/nlp"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/tools"
)

func TestPreprocessor_Process(t *testing.T) {
	pre := nlp.NewPreprocessor(config.NLPConfig{MaxTokens: 512, EnableNormalize: true, EnableStopwords: true}, nil, nopLogger())
	text := "Hello World! This is a test sentence. It has multiple parts."
	r := pre.Process(text)

	if r.WordCount == 0 {
		t.Error("expected non-zero word count")
	}
	if r.CharCount != len(text) {
		t.Errorf("char count: want %d got %d", len(text), r.CharCount)
	}
	if len(r.Tokens) == 0 {
		t.Error("expected tokens")
	}
	if len(r.Sentences) == 0 {
		t.Error("expected sentences")
	}
	if r.Language != "en" {
		t.Errorf("expected en, got %s", r.Language)
	}
}

func TestPreprocessor_Keywords(t *testing.T) {
	pre := nlp.NewPreprocessor(config.NLPConfig{EnableNormalize: true, EnableStopwords: true}, nil, nopLogger())
	kw := pre.Keywords("machine learning algorithms neural networks deep learning artificial intelligence", 3)
	if len(kw) == 0 {
		t.Error("expected keywords")
	}
	t.Logf("keywords: %v", kw)
}

func TestPreprocessor_LanguageDetection(t *testing.T) {
	pre := nlp.NewPreprocessor(config.NLPConfig{}, nil, nopLogger())
	if r := pre.Process("Привет мир"); r.Language != "ru" {
		t.Errorf("expected ru, got %s", r.Language)
	}
	if r := pre.Process("Hello world"); r.Language != "en" {
		t.Errorf("expected en, got %s", r.Language)
	}
}

func TestClassifier_Heuristic(t *testing.T) {
	cls := nlp.NewClassifier(config.NLPConfig{ConfidenceThresh: 0.1}, nil, nopLogger())
	r, err := cls.Classify(context.Background(), "what is the weather today?",
		[]string{"weather query", "calculation", "translation"})
	if err != nil {
		t.Fatalf("Classify error: %v", err)
	}
	if r.Intent != "weather query" {
		t.Errorf("expected weather query, got %s (conf %.2f)", r.Intent, r.Confidence)
	}
}

func TestEntityExtractor_Regex(t *testing.T) {
	ext := nlp.NewEntityExtractor(config.NLPConfig{}, nil, nopLogger())
	entities, err := ext.Extract(context.Background(),
		"Contact info@example.com or visit https://example.com")
	if err != nil {
		t.Fatalf("Extract error: %v", err)
	}
	labels := map[string]bool{}
	for _, e := range entities {
		labels[e.Label] = true
	}
	if !labels["EMAIL"] {
		t.Error("expected EMAIL entity")
	}
	if !labels["URL"] {
		t.Error("expected URL entity")
	}
}

func TestCosineSimilarity(t *testing.T) {
	cases := []struct {
		a, b []float64
		want float64
	}{
		{[]float64{1, 0}, []float64{1, 0}, 1.0},
		{[]float64{1, 0}, []float64{0, 1}, 0.0},
		{[]float64{}, []float64{1}, 0.0},
	}
	for _, c := range cases {
		got := nlp.CosineSimilarity(c.a, c.b)
		diff := got - c.want
		if diff < -0.001 || diff > 0.001 {
			t.Errorf("CosineSimilarity(%v,%v)=%.4f want %.4f", c.a, c.b, got, c.want)
		}
	}
}

func TestCalculatorTool(t *testing.T) {
	calc := &tools.CalculatorTool{}
	cases := []struct{ op, a, b, want string }{
		{"add", "10", "5", "15"},
		{"subtract", "10", "5", "5"},
		{"multiply", "6", "7", "42"},
		{"divide", "10", "4", "2.5"},
		{"power", "2", "10", "1024"},
		{"sqrt", "144", "", "12"},
	}
	for _, c := range cases {
		out, err := calc.Execute(context.Background(), tools.ToolInput{
			Params: map[string]string{"operation": c.op, "a": c.a, "b": c.b},
		})
		if err != nil {
			t.Fatalf("%s: unexpected error: %v", c.op, err)
		}
		if out.IsError {
			t.Errorf("%s: got error output: %s", c.op, out.Content)
			continue
		}
		if out.Content != c.want {
			t.Errorf("%s: want %s, got %s", c.op, c.want, out.Content)
		}
	}
}

func TestCalculatorTool_DivByZero(t *testing.T) {
	calc := &tools.CalculatorTool{}
	out, _ := calc.Execute(context.Background(), tools.ToolInput{
		Params: map[string]string{"operation": "divide", "a": "5", "b": "0"},
	})
	if !out.IsError {
		t.Error("expected error for division by zero")
	}
}

func TestTextSummarizerTool(t *testing.T) {
	s := &tools.TextSummarizerTool{}
	text := "The quick brown fox. It was a good day. The sun was shining. Birds were singing."
	out, err := s.Execute(context.Background(), tools.ToolInput{
		Params: map[string]string{"text": text, "max_sents": "2"},
	})
	if err != nil || out.IsError {
		t.Fatalf("summarizer error: %v / %s", err, out.Content)
	}
	if strings.Count(out.Content, ".") > 2 {
		t.Errorf("expected ≤2 sentences: %s", out.Content)
	}
}

func TestDateTimeTool(t *testing.T) {
	dt := &tools.DateTimeTool{}
	out, err := dt.Execute(context.Background(), tools.ToolInput{
		Params: map[string]string{"timezone": "UTC", "format": "RFC3339"},
	})
	if err != nil || out.IsError || out.Content == "" {
		t.Fatalf("datetime error: %v / %s", err, out.Content)
	}
}

func TestJSONParseTool(t *testing.T) {
	j := &tools.JSONParseTool{}
	out, err := j.Execute(context.Background(), tools.ToolInput{
		Params: map[string]string{
			"json": `{"user":{"name":"Alice"}}`,
			"path": "user.name",
		},
	})
	if err != nil || out.IsError {
		t.Fatalf("json_parse error: %v / %s", err, out.Content)
	}
	if out.Content != `"Alice"` {
		t.Errorf(`expected "Alice", got %s`, out.Content)
	}
}

func TestToolRegistry_Builtins(t *testing.T) {
	reg := tools.NewRegistry()
	for _, name := range []string{"calculator", "text_summarizer", "datetime", "json_parse", "echo"} {
		if _, ok := reg.Get(name); !ok {
			t.Errorf("built-in tool %q not found", name)
		}
	}
}

func BenchmarkPreprocessor(b *testing.B) {
	pre := nlp.NewPreprocessor(config.NLPConfig{EnableNormalize: true, EnableStopwords: true}, nil, nopLogger())
	text := "Machine learning is a subfield of artificial intelligence that focuses on systems that learn from data."
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pre.Process(text)
	}
}

func BenchmarkCalculator(b *testing.B) {
	calc := &tools.CalculatorTool{}
	input := tools.ToolInput{Params: map[string]string{"operation": "multiply", "a": "12345", "b": "67890"}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		calc.Execute(context.Background(), input) //nolint:errcheck
	}
}

func BenchmarkCosineSimilarity(b *testing.B) {
	a, bv := make([]float64, 512), make([]float64, 512)
	for i := range a {
		a[i] = float64(i) / 512.0
		bv[i] = float64(512-i) / 512.0
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nlp.CosineSimilarity(a, bv)
	}
}
