.PHONY: help test test-v bench lint proto docker-build docker-up docker-down clean

# Default target
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Testing
test: ## Run all tests
	go test -race -cover ./...

test-v: ## Run tests with verbose output
	go test -race -cover -v ./...

bench: ## Run benchmarks
	go test -bench=. -benchmem ./...

# Linting
lint: ## Run golangci-lint
	golangci-lint run

# Protocol buffers
proto: ## Generate gRPC stubs from proto files
	protoc --go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		proto/llm/llm.proto

# Docker
docker-build: ## Build Docker image
	docker build -t go-llm-agent-framework .

docker-up: ## Start all services with Docker Compose
	docker compose up --build

docker-down: ## Stop all services
	docker compose down

# Cleanup
clean: ## Clean build artifacts
	go clean
	rm -f main
	rm -f go-llm-agent-framework

# Development
run: ## Run the application locally
	go run main.go

deps: ## Tidy and verify dependencies
	go mod tidy
	go mod verify

# CI simulation
ci: lint test bench docker-build ## Run full CI pipeline locally