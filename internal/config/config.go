package config

import (
	"os"
	"strconv"
	"time"

	"github.com/DennisMRitchie/go-llm-agent-framework/internal/llmclient"
)

// Config holds all configuration for the application
type Config struct {
	Log        LogConfig
	Tracing    TracingConfig
	Agent      AgentConfig
	LLMBackend LLMBackendConfig
	NLP        NLPConfig
	Server     ServerConfig
}

// LogConfig holds logging configuration
type LogConfig struct {
	Level  string
	Format string
}

// TracingConfig holds tracing configuration
type TracingConfig struct {
	Enabled     bool
	ServiceName string
	SampleRate  float64
}

// AgentConfig holds agent configuration
type AgentConfig struct {
	MaxConcurrentTasks int
	Timeout            time.Duration
	MaxMemoryMessages  int
	RateLimitRPS       float64
	RateLimitBurst     int
	MaxWorkers         int
	TaskTimeout        time.Duration
	MaxConcurrentRuns  int
}

// LLMBackendConfig holds LLM backend configuration
type LLMBackendConfig = llmclient.LLMBackendConfig

// NLPConfig holds NLP configuration
type NLPConfig struct {
	Enabled          bool
	EnableNormalize  bool
	EnableStopwords  bool
	MaxTokens        int
	ConfidenceThresh float64
}

// ServerConfig holds server configuration
type ServerConfig struct {
	Port            int
	address         string
	ReadTimeout     time.Duration
	WriteTimeout    time.Duration
	ShutdownTimeout time.Duration
}

// Addr returns the server address
func (s ServerConfig) Addr() string {
	return s.address
}

// Load loads configuration from environment variables
func Load() (*Config, error) {
	cfg := &Config{
		Log: LogConfig{
			Level:  getEnv("LOG_LEVEL", "info"),
			Format: getEnv("LOG_FORMAT", "json"),
		},
		Tracing: TracingConfig{
			Enabled:     getEnvBool("TRACING_ENABLED", false),
			ServiceName: getEnv("TRACING_SERVICE_NAME", "go-llm-agent-framework"),
			SampleRate:  getEnvFloat("TRACING_SAMPLE_RATE", 1.0),
		},
		Agent: AgentConfig{
			MaxConcurrentTasks: getEnvInt("AGENT_MAX_CONCURRENT_TASKS", 10),
			Timeout:            getEnvDuration("AGENT_TIMEOUT", 5*time.Minute),
			MaxMemoryMessages:  getEnvInt("AGENT_MAX_MEMORY_MESSAGES", 100),
			RateLimitRPS:       getEnvFloat("AGENT_RATE_LIMIT_RPS", 10.0),
			RateLimitBurst:     getEnvInt("AGENT_RATE_LIMIT_BURST", 20),
			MaxWorkers:         getEnvInt("AGENT_MAX_WORKERS", 5),
			TaskTimeout:        getEnvDuration("AGENT_TASK_TIMEOUT", 2*time.Minute),
			MaxConcurrentRuns:  getEnvInt("AGENT_MAX_CONCURRENT_RUNS", 10),
		},
		LLMBackend: LLMBackendConfig{
			BaseURL: getEnv("LLM_BACKEND_BASE_URL", "http://localhost:8000"),
			APIKey:  getEnv("LLM_BACKEND_API_KEY", ""),
		},
		NLP: NLPConfig{
			Enabled:          getEnvBool("NLP_ENABLED", true),
			EnableNormalize:  getEnvBool("NLP_ENABLE_NORMALIZE", true),
			EnableStopwords:  getEnvBool("NLP_ENABLE_STOPWORDS", true),
			MaxTokens:        getEnvInt("NLP_MAX_TOKENS", 1000),
			ConfidenceThresh: getEnvFloat("NLP_CONFIDENCE_THRESH", 0.5),
		},
		Server: ServerConfig{
			Port:            getEnvInt("SERVER_PORT", 8080),
			address:         getEnv("SERVER_ADDR", ":8080"),
			ReadTimeout:     getEnvDuration("SERVER_READ_TIMEOUT", 30*time.Second),
			WriteTimeout:    getEnvDuration("SERVER_WRITE_TIMEOUT", 30*time.Second),
			ShutdownTimeout: getEnvDuration("SERVER_SHUTDOWN_TIMEOUT", 30*time.Second),
		},
	}
	return cfg, nil
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if b, err := strconv.ParseBool(value); err == nil {
			return b
		}
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if i, err := strconv.Atoi(value); err == nil {
			return i
		}
	}
	return defaultValue
}

func getEnvDuration(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if d, err := time.ParseDuration(value); err == nil {
			return d
		}
	}
	return defaultValue
}

func getEnvFloat(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		if f, err := strconv.ParseFloat(value, 64); err == nil {
			return f
		}
	}
	return defaultValue
}
