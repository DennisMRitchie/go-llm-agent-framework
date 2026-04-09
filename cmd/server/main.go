package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/DennisMRitchie/go-llm-agent-framework/api"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/agent"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/config"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/llmclient"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/metrics"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/nlp"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/tools"
)

var Version = "dev"

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}

	logger := buildLogger(cfg.Log)
	logger.Info("starting go-llm-agent-framework", "version", Version)

	agentMetrics := metrics.NewAgentMetrics()
	httpMetrics := metrics.NewHTTPMetrics()
	metrics.RegisterAgentMetrics(agentMetrics)
	metrics.RegisterHTTPMetrics(httpMetrics)

	llmClient := llmclient.NewHTTPClient(cfg.LLMBackend, logger)
	logger.Info("LLM backend configured", "url", cfg.LLMBackend.BaseURL)

	toolReg := tools.NewRegistry()
	logger.Info("tools registered", "count", len(toolReg.List()))

	preprocessor := nlp.NewPreprocessor(cfg.NLP, llmClient, logger)
	classifier := nlp.NewClassifier(cfg.NLP, llmClient, logger)
	extractor := nlp.NewEntityExtractor(cfg.NLP, llmClient, logger)

	orchestrator := agent.NewOrchestrator(
		cfg.Agent, llmClient, toolReg,
		preprocessor, classifier, extractor,
		agentMetrics, logger,
	)

	mux := http.NewServeMux()
	handler := api.NewHandler(orchestrator, toolReg, preprocessor, classifier, extractor, httpMetrics, logger)
	handler.Register(mux)

	// Chain middleware
	var h http.Handler = mux
	h = api.MetricsMiddleware(httpMetrics, h)
	h = api.SlogLogger(logger, h)
	h = api.RequestIDMiddleware(h)

	srv := &http.Server{
		Addr:         cfg.Server.Addr(),
		Handler:      h,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
	}

	// Cleanup goroutine
	go func() {
		ticker := time.NewTicker(5 * time.Minute)
		defer ticker.Stop()
		for range ticker.C {
			if n := orchestrator.Cleanup(30 * time.Minute); n > 0 {
				logger.Info("cleaned up old tasks", "count", n)
			}
		}
	}()

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	serverErr := make(chan error, 1)
	go func() {
		logger.Info("HTTP server listening", "addr", cfg.Server.Addr())
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErr <- err
		}
	}()

	select {
	case err := <-serverErr:
		return fmt.Errorf("server error: %w", err)
	case sig := <-quit:
		logger.Info("shutdown signal", "signal", sig.String())
	}

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		logger.Error("shutdown error", "error", err)
	}
	logger.Info("shutdown complete")
	return nil
}

func buildLogger(cfg config.LogConfig) *slog.Logger {
	var level slog.Level
	switch cfg.Level {
	case "debug":
		level = slog.LevelDebug
	case "warn":
		level = slog.LevelWarn
	case "error":
		level = slog.LevelError
	default:
		level = slog.LevelInfo
	}
	opts := &slog.HandlerOptions{Level: level}
	if cfg.Format == "console" {
		return slog.New(slog.NewTextHandler(os.Stdout, opts))
	}
	return slog.New(slog.NewJSONHandler(os.Stdout, opts))
}
