package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/DennisMRitchie/go-llm-agent-framework/api"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/agent"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/config"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/llmclient"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/metrics"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/nlp"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/tools"
)

// Version is set at build time via ldflags.
var Version = "dev"

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	// Load config
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}

	// Logger
	logger, err := buildLogger(cfg.Log)
	if err != nil {
		return fmt.Errorf("build logger: %w", err)
	}
	defer logger.Sync() //nolint:errcheck

	logger.Info("starting go-llm-agent-framework", zap.String("version", Version))

	// Tracing
	var tracerShutdown func(context.Context) error
	if cfg.Tracing.Enabled {
		tp, shutdown, err := initTracer(cfg.Tracing)
		if err != nil {
			return fmt.Errorf("init tracer: %w", err)
		}
		otel.SetTracerProvider(tp)
		tracerShutdown = shutdown
		logger.Info("OpenTelemetry tracing enabled", zap.String("service", cfg.Tracing.ServiceName))
	}

	// Metrics
	agentMetrics := metrics.NewAgentMetrics()
	httpMetrics := metrics.NewHTTPMetrics()

	// LLM client
	llmClient := llmclient.NewHTTPClient(cfg.LLMBackend, logger)
	logger.Info("LLM backend configured", zap.String("url", cfg.LLMBackend.BaseURL))

	// Tool registry
	toolReg := tools.NewRegistry()
	logger.Info("tools registered", zap.Int("count", len(toolReg.List())))

	// NLP pipeline
	preprocessor := nlp.NewPreprocessor(cfg.NLP, llmClient, logger)
	classifier := nlp.NewClassifier(cfg.NLP, llmClient, logger)
	extractor := nlp.NewEntityExtractor(cfg.NLP, llmClient, logger)

	// Agent orchestrator
	orchestrator := agent.NewOrchestrator(
		cfg.Agent,
		llmClient,
		toolReg,
		preprocessor,
		classifier,
		extractor,
		agentMetrics,
		logger,
	)

	// HTTP server
	if cfg.Server.Port != 0 {
		gin.SetMode(gin.ReleaseMode)
	}
	router := gin.New()
	router.Use(gin.Recovery())
	router.Use(api.RequestIDMiddleware())
	router.Use(api.ZapLogger(logger))
	router.Use(api.MetricsMiddleware(httpMetrics))

	handler := api.NewHandler(orchestrator, toolReg, preprocessor, classifier, extractor, httpMetrics, logger)
	handler.Register(router)

	srv := &http.Server{
		Addr:         cfg.Server.Addr(),
		Handler:      router,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
	}

	// Cleanup goroutine
	go func() {
		ticker := time.NewTicker(5 * time.Minute)
		defer ticker.Stop()
		for range ticker.C {
			n := orchestrator.Cleanup(30 * time.Minute)
			if n > 0 {
				logger.Info("cleaned up old tasks", zap.Int("count", n))
			}
		}
	}()

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	serverErr := make(chan error, 1)
	go func() {
		logger.Info("HTTP server listening", zap.String("addr", cfg.Server.Addr()))
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErr <- err
		}
	}()

	select {
	case err := <-serverErr:
		return fmt.Errorf("server error: %w", err)
	case sig := <-quit:
		logger.Info("shutdown signal received", zap.String("signal", sig.String()))
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Error("server shutdown error", zap.Error(err))
	}
	if tracerShutdown != nil {
		if err := tracerShutdown(shutdownCtx); err != nil {
			logger.Error("tracer shutdown error", zap.Error(err))
		}
	}

	logger.Info("shutdown complete")
	return nil
}

func buildLogger(cfg config.LogConfig) (*zap.Logger, error) {
	level, err := zapcore.ParseLevel(cfg.Level)
	if err != nil {
		level = zapcore.InfoLevel
	}

	var zapCfg zap.Config
	if cfg.Format == "console" {
		zapCfg = zap.NewDevelopmentConfig()
	} else {
		zapCfg = zap.NewProductionConfig()
	}
	zapCfg.Level = zap.NewAtomicLevelAt(level)
	return zapCfg.Build()
}

func initTracer(cfg config.TracingConfig) (*sdktrace.TracerProvider, func(context.Context) error, error) {
	exp, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
	if err != nil {
		return nil, nil, fmt.Errorf("create stdout exporter: %w", err)
	}

	res, err := resource.Merge(
		resource.Default(),
		resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceNameKey.String(cfg.ServiceName),
		),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("create resource: %w", err)
	}

	sampler := sdktrace.TraceIDRatioBased(cfg.SampleRate)
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sampler),
	)

	return tp, tp.Shutdown, nil
}
