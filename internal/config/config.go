// Package config loads configuration from environment variables with sensible defaults.
package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

type Config struct {
	Server     ServerConfig
	Agent      AgentConfig
	LLMBackend LLMBackendConfig
	NLP        NLPConfig
	Metrics    MetricsConfig
	Log        LogConfig
}

type ServerConfig struct {
	Host            string
	Port            int
	ReadTimeout     time.Duration
	WriteTimeout    time.Duration
	ShutdownTimeout time.Duration
}

type AgentConfig struct {
	MaxWorkers        int
	TaskTimeout       time.Duration
	MaxMemoryMessages int
	RateLimitRPS      float64
	RateLimitBurst    int
	MaxConcurrentRuns int
}

type LLMBackendConfig struct {
	BaseURL    string
	APIKey     string
	Timeout    time.Duration
	MaxRetries int
	Model      string
	MaxTokens  int
}

type NLPConfig struct {
	MaxTokens        int
	ConfidenceThresh float64
	EnableNormalize  bool
	EnableStopwords  bool
}

type MetricsConfig struct {
	Enabled bool
	Path    string
}

type LogConfig struct {
	Level  string
	Format string
}

func Load() (*Config, error) {
	return &Config{
		Server: ServerConfig{
			Host:            envStr("SERVER_HOST", "0.0.0.0"),
			Port:            envInt("SERVER_PORT", 8080),
			ReadTimeout:     envDur("SERVER_READ_TIMEOUT", 30*time.Second),
			WriteTimeout:    envDur("SERVER_WRITE_TIMEOUT", 30*time.Second),
			ShutdownTimeout: envDur("SERVER_SHUTDOWN_TIMEOUT", 10*time.Second),
		},
		Agent: AgentConfig{
			MaxWorkers:        envInt("AGENT_MAX_WORKERS", 10),
			TaskTimeout:       envDur("AGENT_TASK_TIMEOUT", 120*time.Second),
			MaxMemoryMessages: envInt("AGENT_MAX_MEMORY_MESSAGES", 50),
			RateLimitRPS:      envFloat("AGENT_RATE_LIMIT_RPS", 10.0),
			RateLimitBurst:    envInt("AGENT_RATE_LIMIT_BURST", 20),
			MaxConcurrentRuns: envInt("AGENT_MAX_CONCURRENT_RUNS", 5),
		},
		LLMBackend: LLMBackendConfig{
			BaseURL:    envStr("LLM_BASE_URL", "http://localhost:8000"),
			APIKey:     envStr("LLM_API_KEY", ""),
			Timeout:    envDur("LLM_TIMEOUT", 60*time.Second),
			MaxRetries: envInt("LLM_MAX_RETRIES", 3),
			Model:      envStr("LLM_MODEL", "gpt-3.5-turbo"),
			MaxTokens:  envInt("LLM_MAX_TOKENS", 1024),
		},
		NLP: NLPConfig{
			MaxTokens:        envInt("NLP_MAX_TOKENS", 512),
			ConfidenceThresh: envFloat("NLP_CONFIDENCE_THRESHOLD", 0.6),
			EnableNormalize:  envBool("NLP_ENABLE_NORMALIZE", true),
			EnableStopwords:  envBool("NLP_ENABLE_STOPWORDS", true),
		},
		Metrics: MetricsConfig{
			Enabled: envBool("METRICS_ENABLED", true),
			Path:    envStr("METRICS_PATH", "/metrics"),
		},
		Log: LogConfig{
			Level:  envStr("LOG_LEVEL", "info"),
			Format: envStr("LOG_FORMAT", "json"),
		},
	}, nil
}

func (s ServerConfig) Addr() string { return fmt.Sprintf("%s:%d", s.Host, s.Port) }

func envStr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func envBool(key string, def bool) bool {
	if v := os.Getenv(key); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return def
}

func envDur(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}
