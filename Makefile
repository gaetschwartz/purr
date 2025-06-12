# Whisper UI Makefile

.PHONY: help build build-release test test-verbose clean install dev lint format check setup-models

# Default target
help:
	@echo "Whisper UI Development Commands:"
	@echo ""
	@echo "Building:"
	@echo "  build         - Build debug version"
	@echo "  build-release - Build optimized release version"
	@echo ""
	@echo "Testing:"
	@echo "  test          - Run all tests"
	@echo "  test-verbose  - Run tests with verbose output"
	@echo "  test-core     - Run only core library tests"
	@echo "  test-cli      - Run only CLI tests"
	@echo ""
	@echo "Development:"
	@echo "  dev           - Run in development mode with file watching"
	@echo "  lint          - Run clippy linter"
	@echo "  format        - Format code with rustfmt"
	@echo "  check         - Check code without building"
	@echo ""
	@echo "Setup:"
	@echo "  install       - Install binary to cargo bin directory"
	@echo "  setup-models  - Download Whisper models"
	@echo "  clean         - Clean build artifacts"

# Building
build:
	cargo build

build-release:
	cargo build --release

# Testing
test:
	cargo test

test-verbose:
	cargo test -- --nocapture

test-core:
	cargo test -p whisper-ui-core

test-cli:
	cargo test -p whisper-ui-cli

# Development
dev:
	cargo watch -x "build" -x "test"

lint:
	cargo clippy -- -D warnings

format:
	cargo fmt

check:
	cargo check

# Installation and setup
install:
	cargo install --path whisper-ui-cli

setup-models:
	@echo "Creating models directory..."
	@mkdir -p models
	@echo "Downloading base English model (142 MB)..."
	@curl -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin" -o models/ggml-base.en.bin
	@echo "Downloading tiny English model (39 MB)..."
	@curl -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin" -o models/ggml-tiny.en.bin
	@echo "Models downloaded successfully!"

clean:
	cargo clean

# Platform-specific commands
setup-windows:
	@echo "Setting up Windows development environment..."
	@echo "Please install FFmpeg manually from https://ffmpeg.org/download.html"
	@echo "Or use: choco install ffmpeg"

setup-linux:
	sudo apt update
	sudo apt install ffmpeg libavformat-dev libavcodec-dev libavutil-dev libswresample-dev pkg-config

setup-macos:
	brew install ffmpeg pkg-config

# CI/CD commands
ci-test:
	cargo test --release

ci-build:
	cargo build --release

ci-lint:
	cargo clippy -- -D warnings
	cargo fmt -- --check

# Benchmarking (if you want to add benchmarks later)
bench:
	cargo bench

# Documentation
docs:
	cargo doc --open

# Example usage
example:
	@echo "Example usage:"
	@echo "1. Build the project: make build-release"
	@echo "2. Download models: make setup-models" 
	@echo "3. Run transcription: ./target/release/whisper-ui-cli audio.wav"

# Development workflow
dev-setup: setup-models build test
	@echo "Development environment ready!"

# Release workflow  
release: clean lint test build-release
	@echo "Release build completed successfully!"
