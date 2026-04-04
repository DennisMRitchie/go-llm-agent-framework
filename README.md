# go-llm-agent-framework

[![CI](https://github.com/DennisMRitchie/go-llm-agent-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/DennisMRitchie/go-llm-agent-framework/actions/workflows/ci.yml)
[![Go Version](https://img.shields.io/badge/Go-1.22-00ADD8?logo=go)](https://golang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/DennisMRitchie/go-llm-agent-framework)](https://goreportcard.com/report/github.com/DennisMRitchie/go-llm-agent-framework)
[![codecov](https://codecov.io/gh/DennisMRitchie/go-llm-agent-framework/branch/main/graph/badge.svg)](https://codecov.io/gh/DennisMRitchie/go-llm-agent-framework)

A **lightweight, production-ready Go framework** for building LLM-powered agents with NLP capabilities. Implements the ReAct (Reason + Act) pattern with parallel tool execution, sliding-window memory, cosine-similarity search, and a FastAPI Python sidecar for LLM/NLP inference.

Built to demonstrate advanced Go patterns used in real AI infrastructure: worker pools, context propagation, structured concurrency, OpenTelemetry tracing, and Prometheus metrics — all wired together cleanly.

---

## ✨ Features

| Feature | Details |
|---|---|
| **ReAct agent loop** | Reason → Act → Observe cycles with configurable max iterations |
| **Parallel tool execution** | Worker-pool goroutines execute multiple tool calls concurrently per turn |
| **Conversation memory** | Sliding-window history with cosine-similarity semantic search |
| **Built-in NLP pipeline** | Normalization, tokenization, stop-word filtering, language detection |
| **Intent classification** | Heuristic fallback + optional zero-shot via HuggingFace BART |
| **Entity extraction** | Regex fallback + optional HuggingFace BERT NER |
| **Pluggable tool registry** | Register any `Tool` interface implementation at runtime |
| **Rate limiting** | Token-bucket per agent via `golang.org/x/time/rate` |
| **OpenTelemetry tracing** | Spans across agent runs, tool calls, and LLM requests |
| **Prometheus metrics** | Task counters, duration histograms, token usage, queue depth |
| **Graceful shutdown** | Full signal handling with configurable drain timeout |
| **Docker Compose stack** | Agent + Python LLM + Prometheus + Grafana — one command |
| **gRPC ready** | Proto definition included; run `make proto` to generate stubs |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HTTP Client / curl                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ REST API (Gin)
┌──────────────────────────────▼──────────────────────────────────────┐
│                         api/handler.go                               │
│   POST /api/v1/agent/run          GET  /api/v1/agent/tasks/:id       │
│   POST /api/v1/agent/run/sync     POST /api/v1/nlp/classify          │
│   POST /api/v1/nlp/process        POST /api/v1/nlp/entities          │
│   GET  /api/v1/tools              GET  /metrics   GET /health        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                    agent/orchestrator.go                             │
│                                                                      │
│   Submit()  ──► goroutine pool (MaxConcurrentRuns semaphore)         │
│   Wait()    ◄── taskEntry{status, result, cancel, done chan}         │
│   Status()  Cancel()  ListTasks()  Cleanup()                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ one Agent per task
┌──────────────────────────────▼──────────────────────────────────────┐
│                       agent/agent.go                                 │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  ReAct Loop (max 5 iterations)                               │  │
│   │                                                              │  │
│   │  NLP preprocess ──► LLM Complete ──► parse <tool_call>      │  │
│   │       │                                     │               │  │
│   │  memory.Add()              ┌────────────────┴─────────┐     │  │
│   │                            │  Parallel Tool Execution │     │  │
│   │                            │  (worker pool goroutines)│     │  │
│   │                            │  tool1 ── tool2 ── toolN │     │  │
│   │                            └────────────┬─────────────┘     │  │
│   │  memory.Add(results) ◄─────────────────┘                    │  │
│   └──────────────────────────────────────────────────────────────┘  │
└────────────┬──────────────────────────────────────┬─────────────────┘
             │                                       │
┌────────────▼──────────────┐          ┌────────────▼────────────────┐
│    agent/memory.go         │          │    internal/tools/          │
│                            │          │                             │
│  entries []*MemoryEntry    │          │  Registry (sync.RWMutex)    │
│  sliding-window eviction   │          │  calculator                 │
│  cosine similarity search  │          │  text_summarizer            │
│  lazy embedding cache      │          │  datetime                   │
└────────────────────────────┘          │  json_parse                 │
                                        │  echo                       │
┌───────────────────────────┐           │  + your custom tools        │
│    internal/nlp/          │           └────────────────────────────┘
│                           │
│  Preprocessor             │          ┌────────────────────────────┐
│    normalize / tokenize   │          │  Python LLM Backend        │
│    sentence split         │◄────────►│  (FastAPI)                 │
│    keyword extraction     │  HTTP    │                             │
│    TF embedding           │  or gRPC │  POST /v1/complete         │
│                           │          │  POST /v1/classify          │
│  Classifier               │          │  POST /v1/entities          │
│    zero-shot NLI          │          │                             │
│    heuristic fallback     │          │  Backends:                  │
│                           │          │    OpenAI API               │
│  EntityExtractor          │          │    HuggingFace models       │
│    BERT NER               │          │    Stub (no key needed)     │
│    regex fallback         │          └────────────────────────────┘
└───────────────────────────┘
```

---

## Quick Start

### Option 1 — Docker Compose (recommended)

```bash
git clone https://github.com/DennisMRitchie/go-llm-agent-framework
cd go-llm-agent-framework

# Optional: add your OpenAI key for real completions
echo "OPENAI_API_KEY=sk-..." > .env

docker compose up --build
```

Services:
- Go agent API → `http://localhost:8080`
- Python LLM backend → `http://localhost:8000`
- Prometheus → `http://localhost:9090`
- Grafana → `http://localhost:3000` (admin / admin)

### Option 2 — Local Go (stub mode, no Python needed)

```bash
git clone https://github.com/DennisMRitchie/go-llm-agent-framework
cd go-llm-agent-framework

go mod tidy
go run ./cmd/server
```

The agent starts in stub mode — the Python backend is optional. All NLP fallbacks and tool executions work without it.

---

## API Reference

### Run a task (async)

```bash
curl -s -X POST http://localhost:8080/api/v1/agent/run \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "What is 2 to the power of 10?",
    "session_id": "demo-session"
  }'
```

```json
{
  "task_id": "3f2a1b4c-...",
  "status": "pending",
  "created_at": "2026-03-31T12:00:00Z"
}
```

### Poll task status

```bash
curl http://localhost:8080/api/v1/agent/tasks/3f2a1b4c-...
```

### Run synchronously (blocking)

```bash
curl -s -X POST http://localhost:8080/api/v1/agent/run/sync \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "What is 6 multiplied by 7?", "session_id": "s1"}' | jq .
```

```json
{
  "task_id": "...",
  "status": "completed",
  "response": "The result is 42.",
  "tool_calls": [
    {
      "tool_name": "calculator",
      "params": {"operation": "multiply", "a": "6", "b": "7"},
      "output": "42",
      "is_error": false,
      "duration_ms": 120000
    }
  ],
  "tokens_used": 34,
  "duration_ms": 280
}
```

### NLP endpoints

```bash
# Preprocess text
curl -s -X POST http://localhost:8080/api/v1/nlp/process \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello World! Machine learning is amazing."}'

# Intent classification
curl -s -X POST http://localhost:8080/api/v1/nlp/classify \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "What is the weather in Paris today?",
    "labels": ["weather query", "calculation", "translation", "general question"]
  }'

# Entity extraction
curl -s -X POST http://localhost:8080/api/v1/nlp/entities \
  -H 'Content-Type: application/json' \
  -d '{"text": "Contact support at help@example.com or visit https://docs.example.com"}'

# Top keywords
curl -s -X POST http://localhost:8080/api/v1/nlp/keywords \
  -H 'Content-Type: application/json' \
  -d '{"text": "Deep learning neural networks outperform traditional machine learning on vision tasks", "top_n": 5}'

# List registered tools
curl http://localhost:8080/api/v1/tools
```

---

## Adding a Custom Tool

```go
package tools

import (
    "context"
    "github.com/DennisMRitchie/go-llm-agent-framework/internal/tools"
)

type WeatherTool struct{ APIKey string }

func (w *WeatherTool) Name()        string { return "weather" }
func (w *WeatherTool) Description() string { return "Gets current weather for a city." }
func (w *WeatherTool) Schema() map[string]any {
    return map[string]any{
        "type": "object",
        "properties": map[string]any{
            "city": map[string]any{"type": "string"},
        },
        "required": []string{"city"},
    }
}

func (w *WeatherTool) Execute(ctx context.Context, input tools.ToolInput) (*tools.ToolOutput, error) {
    city := input.Params["city"]
    // ... call weather API ...
    return &tools.ToolOutput{Content: "Sunny, 22°C in " + city}, nil
}
```

Then register it:
```go
toolReg.Register(&WeatherTool{APIKey: os.Getenv("WEATHER_API_KEY")})
```

---

## Configuration

All settings can be overridden via environment variables using the `AGENT_` prefix with `_` replacing `.`:

| Config key | Env var | Default |
|---|---|---|
| `server.port` | `AGENT_SERVER_PORT` | `8080` |
| `llm_backend.base_url` | `AGENT_LLM_BACKEND_BASE_URL` | `http://python-llm:8000` |
| `llm_backend.api_key` | `AGENT_LLM_BACKEND_API_KEY` | `` |
| `agent.max_workers` | `AGENT_AGENT_MAX_WORKERS` | `10` |
| `agent.task_timeout` | `AGENT_AGENT_TASK_TIMEOUT` | `120s` |
| `agent.rate_limit_rps` | `AGENT_AGENT_RATE_LIMIT_RPS` | `10.0` |
| `agent.max_concurrent_runs` | `AGENT_AGENT_MAX_CONCURRENT_RUNS` | `5` |
| `log.level` | `AGENT_LOG_LEVEL` | `info` |
| `tracing.enabled` | `AGENT_TRACING_ENABLED` | `true` |

See [`config.yaml`](config.yaml) for the full reference.

---

## Python Backend Modes

The Python sidecar (`python/server.py`) supports three modes:

| Mode | How to enable | Notes |
|---|---|---|
| **Stub** | Default (no env vars) | Returns mock responses, good for demos |
| **OpenAI** | `OPENAI_API_KEY=sk-...` | Uses `gpt-3.5-turbo` by default |
| **HuggingFace** | `USE_HF=1` | Downloads BART (NLI) + BERT (NER) on first run |

```bash
# OpenAI mode
OPENAI_API_KEY=sk-... python python/server.py

# HuggingFace mode (local, no API key)
USE_HF=1 python python/server.py

# Custom model
HF_NLI_MODEL=cross-encoder/nli-MiniLM2-L6-H768 USE_HF=1 python python/server.py
```

---

## Development

```bash
# Run all tests with race detector
make test

# Verbose test output
make test-v

# Run benchmarks
make bench

# Lint
make lint

# Generate gRPC stubs (requires protoc)
make proto

# Full stack
make docker-up
make docker-down
```

### Benchmark results (Apple M2, Go 1.22)

```
BenchmarkPreprocessor_Process-8     312451    3821 ns/op    2048 B/op    31 allocs/op
BenchmarkCalculator-8              4823112     249 ns/op      96 B/op     4 allocs/op
BenchmarkCosineSimilarity-8        1000000    1203 ns/op    8192 B/op     2 allocs/op
```

---

## Project Structure

```
go-llm-agent-framework/
├── cmd/server/main.go              # Entrypoint, wiring, graceful shutdown
├── api/handler.go                  # Gin HTTP handlers + middleware
├── internal/
│   ├── agent/
│   │   ├── agent.go               # ReAct loop, parallel tool execution
│   │   ├── orchestrator.go        # Concurrent task pool, status tracking
│   │   └── memory.go              # Sliding-window + similarity search
│   ├── config/config.go           # Viper config with env override
│   ├── llmclient/client.go        # HTTP client + MockClient for testing
│   ├── metrics/metrics.go         # Prometheus metric definitions
│   ├── nlp/nlp.go                 # NLP pipeline, classifier, extractor
│   └── tools/tools.go             # Tool interface, registry, built-ins
├── proto/llm/llm.proto            # gRPC service definition
├── python/
│   ├── server.py                  # FastAPI LLM backend
│   ├── requirements.txt
│   └── Dockerfile.python
├── tests/
│   ├── agent_test.go              # Agent, orchestrator, memory tests
│   └── nlp_test.go                # NLP, classifier, tools, benchmarks
├── deploy/prometheus.yml
├── config.yaml
├── docker-compose.yml
├── Dockerfile
└── Makefile
```

---

## Related Projects

This framework is part of a Go LLM infrastructure portfolio:

| Repository | Description |
|---|---|
| [go-llm-agent-framework](https://github.com/DennisMRitchie/go-llm-agent-framework) | ← This repo |
| [go-rag-guardrails](https://github.com/DennisMRitchie/go-rag-guardrails) | Security middleware: prompt injection, PII redaction, toxicity |
| [go-llm-evaluator](https://github.com/DennisMRitchie/go-llm-evaluator) | LLM-as-a-Judge evaluation framework |
| [go-llm-smart-cache](https://github.com/DennisMRitchie/go-llm-smart-cache) | Semantic caching with Redis + OpenTelemetry |
| [go-semantic-chunker](https://github.com/DennisMRitchie/go-semantic-chunker) | Semantic text chunking for RAG |
| [go-rag-llm-orchestrator](https://github.com/DennisMRitchie/go-rag-llm-orchestrator) | Full RAG orchestrator with gRPC + Python sidecar |

---

## License

MIT — see [LICENSE](LICENSE).
