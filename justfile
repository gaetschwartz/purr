# Justfile for Purr WebGPU workspace test management

# Default recipe
default:
    @just --list

# Run all tests with coverage
test:
    ./scripts/test-runner.sh

# Run tests without coverage
test-fast:
    ./scripts/test-runner.sh --no-coverage

# Run tests for specific crate
test-crate crate:
    ./scripts/test-runner.sh --crate {{crate}}

# Run unit tests only
test-unit:
    cargo test --workspace --lib --profile test

# Run integration tests only
test-integration:
    cargo test --workspace --test '*' --profile test

# Run documentation tests
test-doc:
    cargo test --workspace --doc --profile test

# Run WASM tests
test-wasm:
    wasm-pack test --node crates/purr-ui --features web
    wasm-pack test --node crates/purr-web

# Run benchmarks
bench:
    ./scripts/test-runner.sh --benchmarks

# Check for flaky tests
test-flaky:
    ./scripts/test-runner.sh --flakiness

# Generate coverage report only
coverage:
    cargo tarpaulin --workspace --out Html Xml Json Lcov --output-dir target/coverage --timeout 300 --exclude purr-test-utils

# Open coverage report in browser
coverage-open:
    @just coverage
    @open target/coverage/tarpaulin-report.html || xdg-open target/coverage/tarpaulin-report.html

# Run linting
lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Format code
fmt:
    cargo fmt --all

# Check formatting
fmt-check:
    cargo fmt --all -- --check

# Run all quality checks
check-all:
    @just fmt-check
    @just lint
    @just test-fast

# Clean all build artifacts
clean:
    cargo clean
    rm -rf target/coverage target/test-reports

# Install development dependencies
install-deps:
    cargo install cargo-tarpaulin wasm-pack cargo-nextest
    rustup target add wasm32-unknown-unknown

# Setup development environment
setup:
    @just install-deps
    mkdir -p target/coverage target/test-reports
    echo "Development environment setup complete!"

# Run tests with nextest (if available)
test-nextest:
    #!/usr/bin/env bash
    if command -v cargo-nextest &> /dev/null; then
        cargo nextest run --workspace --profile test
    else
        echo "cargo-nextest not installed, falling back to standard test runner"
        cargo test --workspace --profile test
    fi

# Profile test performance
profile-tests:
    cargo test --workspace --profile test-opt --release

# Validate workspace configuration
validate:
    cargo check --workspace --all-targets --all-features
    cargo tree --workspace --duplicates
    cargo audit || echo "cargo-audit not installed, skipping security check"

# Generate test documentation
docs:
    cargo doc --workspace --document-private-items --open

# Run stress tests
stress-test:
    #!/usr/bin/env bash
    echo "Running stress tests..."
    for i in {1..10}; do
        echo "Stress test iteration $i/10"
        cargo test --workspace --profile test-opt || exit 1
    done
    echo "Stress tests completed successfully!"

# Watch mode for continuous testing
watch:
    #!/usr/bin/env bash
    if command -v cargo-watch &> /dev/null; then
        cargo watch -x "test --workspace --lib"
    else
        echo "cargo-watch not installed. Install with: cargo install cargo-watch"
        exit 1
    fi

# Memory profiling (requires valgrind on Linux)
profile-memory:
    #!/usr/bin/env bash
    if command -v valgrind &> /dev/null; then
        valgrind --tool=massif --stacks=yes cargo test --workspace --profile test-opt
    else
        echo "valgrind not available, skipping memory profiling"
    fi

# Security audit
audit:
    #!/usr/bin/env bash
    if command -v cargo-audit &> /dev/null; then
        cargo audit
    else
        echo "Installing cargo-audit..."
        cargo install cargo-audit
        cargo audit
    fi

# Update dependencies
update:
    cargo update --workspace
    @just validate

# Pre-commit checks
pre-commit:
    @just fmt-check
    @just lint
    @just test-fast
    @just audit

# CI simulation
ci:
    @echo "Simulating CI pipeline..."
    @just clean
    @just validate
    @just pre-commit
    @just test
    @echo "CI simulation completed successfully!"

# Create release build with tests
release:
    @just clean
    @just ci
    cargo build --release --workspace
    @echo "Release build completed successfully!"

fetch-model model *ARGS:
    #!/usr/bin/env bash
    MODEL="ggml-{{model}}"
    MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${MODEL}.bin"
    DEST_DIR="./.local/models"
    mkdir -p $DEST_DIR
    # check if '-f' flag is passed to force re-download
    FORCE_DOWNLOAD=false
    for arg in "{{ARGS}}"; do
        if [ "$arg" == "-f" ] || [ "$arg" == "--force" ]; then
            FORCE_DOWNLOAD=true
        fi
    done
    if [ ! -f "$DEST_DIR/${MODEL}.bin" ] || [ "$FORCE_DOWNLOAD" = true ]; then
        echo "Downloading model ${MODEL}..."
        TMP_DIR=$(mktemp -d)
        # use wget to download the model, fail if the download fails
        wget -q --show-progress -O "$TMP_DIR/${MODEL}.bin" $MODEL_URL || { echo "Download failed!"; rm -rf $TMP_DIR; exit 1; }
        mv "$TMP_DIR/${MODEL}.bin" "$DEST_DIR/${MODEL}.bin"
        rm -rf $TMP_DIR
        echo "Model ${MODEL} downloaded to $DEST_DIR"
    else
        echo "Model ${MODEL} already exists in $DEST_DIR, use -f to force re-download"
    fi