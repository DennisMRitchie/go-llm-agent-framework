package tests

import (
	"context"
	"testing"

	"github.com/DennisMRitchie/go-llm-agent-framework/internal/config"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/nlp"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/tools"
	"go.uber.org/zap/zaptest"
)

func TestPreprocessor_Process(t *testing.T) {
	logger := zaptest.NewLogger(t)
	cfg := config.NLPConfig{
		EnableNormalize: true,
		EnableStopwords: true,
		MaxTokens:       100,
	}

	preprocessor := nlp.NewPreprocessor(cfg, nil, logger)

	text := "Hello World! This is a TEST."
	result := preprocessor.Process(text)

	if len(result.Tokens) == 0 {
		t.Error("Expected tokens to be generated")
	}

	if len(result.Sentences) == 0 {
		t.Error("Expected sentences to be split")
	}
}

func BenchmarkPreprocessor_Process(b *testing.B) {
	logger := zaptest.NewLogger(b)
	cfg := config.NLPConfig{
		EnableNormalize: true,
		EnableStopwords: true,
		MaxTokens:       1000,
	}

	preprocessor := nlp.NewPreprocessor(cfg, nil, logger)
	text := "This is a sample text for benchmarking the NLP preprocessor. It contains multiple sentences and various words that need to be processed."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		preprocessor.Process(text)
	}
}

func BenchmarkCalculator(b *testing.B) {
	calc := &tools.CalculatorTool{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = calc.Execute(context.Background(), tools.ToolInput{
			Params: map[string]string{
				"operation": "add",
				"a":         "10",
				"b":         "20",
			},
		})
	}
}
