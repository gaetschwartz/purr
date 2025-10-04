# Default settings
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
export WINDOWS_TEST_ARGS := env("WINDOWS_TEST_ARGS", "--features cuda")

# Colors for output
RED := '\033[0;31m'
GREEN := '\033[0;32m'
YELLOW := '\033[1;33m'
BLUE := '\033[0;34m'
NC := '\033[0m'

# Test configuration
test_profile := "test"
timeout_seconds := "300"

# Run all tests using cargo nextest
[macos]
test *ARGS:
    @echo "{{BLUE}}[INFO]{{NC}} Running tests with nextest..."
    cargo nextest run --features metal {{ARGS}}

[windows]
test *ARGS:
    @echo "{{BLUE}}[INFO]{{NC}} Running tests with nextest..."
    cargo nextest run $WINDOWS_TEST_ARGS {{ARGS}}

# Run tests for a specific crate
test-crate crate *ARGS:
    @echo "{{BLUE}}[INFO]{{NC}} Testing crate: {{crate}}"
    cargo nextest run --package {{crate}} {{ARGS}}

# Run tests with verbose output
test-verbose:
    @echo "{{BLUE}}[INFO]{{NC}} Running tests with verbose output..."
    cargo nextest run --verbose

# Run unit tests only
test-unit:
    @echo "{{BLUE}}[INFO]{{NC}} Running unit tests..."
    cargo nextest run --lib

# Run integration tests only
test-integration:
    @echo "{{BLUE}}[INFO]{{NC}} Running integration tests..."
    cargo nextest run --test '*'

# Run doctests
test-doc:
    @echo "{{BLUE}}[INFO]{{NC}} Running documentation tests..."
    cargo test --doc --workspace

# Generate coverage report with tarpaulin
coverage *ARGS:
    @echo "{{BLUE}}[INFO]{{NC}} Generating coverage report..."
    cargo tarpaulin --out Html Xml --output-dir target/coverage --timeout {{timeout_seconds}} --workspace {{ARGS}}

# Run benchmarks
bench *ARGS:
    @echo "{{BLUE}}[INFO]{{NC}} Running benchmarks..."
    cargo bench {{ARGS}}

# Check for flaky tests by running them multiple times
test-flaky runs="5":
    @echo "{{BLUE}}[INFO]{{NC}} Checking for flaky tests ({{runs}} runs)..."
    @for i in $(seq 1 {{runs}}); do \
        echo -e "{{BLUE}}[INFO]{{NC}} Run $$i/{{runs}}"; \
        cargo nextest run --no-fail-fast || true; \
    done

# Run WASM tests (if wasm-pack is available)
test-wasm:
    #!/usr/bin/env bash
    echo -e "{{BLUE}}[INFO]{{NC}} Running WASM tests..."
    if ! command -v wasm-pack &> /dev/null; then
        echo -e "{{YELLOW}}[WARNING]{{NC}} wasm-pack not found, skipping WASM tests"
        exit 0
    fi

    declare -A WASM_CRATES
    WASM_CRATES=(
        #["purr-app"]="--features web"
        ["purr-web"]="--chrome --headless"
    )
    for crate in "${!WASM_CRATES[@]}"; do
        if [[ -d "crates/$crate" ]]; then
            echo -e "{{BLUE}}[INFO]{{NC}} Testing WASM for $crate..."
            pushd "crates/$crate" > /dev/null
            if grep -q "wasm-bindgen-test" Cargo.toml 2>/dev/null; then
                wasm-pack test ${WASM_CRATES[$crate]} || echo -e "{{YELLOW}}[WARNING]{{NC}} WASM tests failed for $crate"
            fi
            popd > /dev/null
        fi
    done

# Clean test artifacts and reports
clean-tests:
    @echo "{{BLUE}}[INFO]{{NC}} Cleaning test artifacts..."
    rm -rf target/coverage target/test-reports target/nextest

# Install test dependencies
install-test-deps:
    @echo "{{BLUE}}[INFO]{{NC}} Installing test dependencies..."
    cargo install cargo-nextest --locked
    cargo install cargo-tarpaulin --locked

# Run all test types in sequence
test-all: test test-doc test-wasm
    @echo "{{GREEN}}[SUCCESS]{{NC}} All tests completed!"

# Run tests with coverage and save report
test-with-coverage: test coverage
    @echo "{{GREEN}}[SUCCESS]{{NC}} Tests with coverage completed!"
    @echo "Coverage reports available in target/coverage/"

# Run tests with specific profile (test, test-opt, dev, release)
test-profile profile *ARGS:
    @echo "{{BLUE}}[INFO]{{NC}} Running tests with profile: {{profile}}"
    cargo nextest run --cargo-profile {{profile}} {{ARGS}}

# Quick test for CI/CD (fail fast, no capture)
test-ci:
    @echo "{{BLUE}}[INFO]{{NC}} Running CI tests (fail-fast mode)..."
    cargo nextest run --fail-fast --status-level fail

# Run tests with timeout
test-timeout seconds="60" *ARGS:
    @echo "{{BLUE}}[INFO]{{NC}} Running tests with {{seconds}}s timeout per test..."
    timeout {{seconds}} cargo nextest run {{ARGS}} || echo -e "{{YELLOW}}[WARNING]{{NC}} Tests timed out after {{seconds}} seconds"

# List all available tests
test-list *ARGS:
    @echo "{{BLUE}}[INFO]{{NC}} Listing available tests..."
    cargo nextest list {{ARGS}}

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

trigger-nightly:
    @echo "{{BLUE}}[INFO]{{NC}} Triggering nightly build workflow..."
    gh workflow run build.yml -f nightly=true
    @echo "{{GREEN}}[SUCCESS]{{NC}} Nightly build triggered!"

[macos]
_cn *ARGS:
    @echo "{{BLUE}}[INFO]{{NC}} Running tests with Metal support..."
    cargo nextest run --features metal {{ARGS}}

[windows]
dx-build *ARGS:
    $env:CARGO_TARGET_DIR="{{invocation_directory()}}\t" ; \
    dx build -p purr-app --release --desktop --profile r {{ARGS}}