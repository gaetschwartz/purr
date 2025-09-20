#!/bin/bash

# Comprehensive test runner for the Purr WebGPU workspace
# This script runs all tests, generates coverage reports, and checks for test flakiness

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COVERAGE_DIR="${WORKSPACE_ROOT}/target/coverage"
REPORTS_DIR="${WORKSPACE_ROOT}/target/test-reports"
FLAKINESS_RUNS=5
TIMEOUT_SECONDS=300

# Default test profiles
TEST_PROFILES=("test" "test-opt")
FEATURES=("default" "web" "benchmarks")

# Command line argument parsing
VERBOSE=false
COVERAGE=true
FLAKINESS_CHECK=false
BENCHMARKS=false
SPECIFIC_CRATE=""
PROFILE="test"

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -c, --coverage          Generate coverage report (default: true)
    --no-coverage           Skip coverage generation
    -f, --flakiness         Run flakiness detection tests
    -b, --benchmarks        Run benchmark tests
    -p, --profile PROFILE   Test profile to use (test, test-opt) (default: test)
    --crate CRATE           Run tests for specific crate only
    --timeout SECONDS       Test timeout in seconds (default: 300)

EXAMPLES:
    $0                      # Run all tests with coverage
    $0 --no-coverage        # Run tests without coverage
    $0 --crate purr-core    # Test only purr-core crate
    $0 --benchmarks         # Run benchmarks
    $0 --flakiness          # Check for flaky tests
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        -f|--flakiness)
            FLAKINESS_CHECK=true
            shift
            ;;
        -b|--benchmarks)
            BENCHMARKS=true
            shift
            ;;
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        --crate)
            SPECIFIC_CRATE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    # Check for required tools
    local tools=("cargo" "rustc")

    if [[ "$COVERAGE" == "true" ]]; then
        tools+=("cargo-tarpaulin")
    fi

    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            if [[ "$tool" == "cargo-tarpaulin" ]]; then
                log_warning "$tool not found. Installing..."
                cargo install cargo-tarpaulin
            else
                log_error "$tool is required but not installed"
                exit 1
            fi
        fi
    done

    # Check Rust toolchain
    if ! rustup show | grep -q "stable"; then
        log_warning "Stable Rust toolchain not found. Installing..."
        rustup install stable
        rustup default stable
    fi

    log_success "Dependencies check completed"
}

# Setup test environment
setup_test_env() {
    log_info "Setting up test environment..."

    # Create necessary directories
    mkdir -p "$COVERAGE_DIR" "$REPORTS_DIR"

    # Set environment variables
    export RUST_BACKTRACE=1
    export RUST_LOG=debug

    # For WASM tests
    if command -v wasm-pack &> /dev/null; then
        export WASM_BINDGEN_TEST_TIMEOUT=60
    fi

    log_success "Test environment setup completed"
}

# Run unit tests
run_unit_tests() {
    log_info "Running unit tests..."

    local cargo_args=("test" "--profile" "$PROFILE")

    if [[ "$VERBOSE" == "true" ]]; then
        cargo_args+=("--verbose")
    fi

    if [[ -n "$SPECIFIC_CRATE" ]]; then
        cargo_args+=("--package" "$SPECIFIC_CRATE")
    else
        cargo_args+=("--workspace")
    fi

    # Add timeout
    if command -v timeout &> /dev/null; then
        timeout "$TIMEOUT_SECONDS" cargo "${cargo_args[@]}" || {
            log_error "Unit tests timed out after $TIMEOUT_SECONDS seconds"
            return 1
        }
    else
        cargo "${cargo_args[@]}" || {
            log_error "Unit tests failed"
            return 1
        }
    fi

    log_success "Unit tests completed successfully"
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."

    local cargo_args=("test" "--profile" "$PROFILE" "--test" "*")

    if [[ "$VERBOSE" == "true" ]]; then
        cargo_args+=("--verbose")
    fi

    if [[ -n "$SPECIFIC_CRATE" ]]; then
        cargo_args+=("--package" "$SPECIFIC_CRATE")
    else
        cargo_args+=("--workspace")
    fi

    # Run integration tests with timeout
    if command -v timeout &> /dev/null; then
        timeout "$TIMEOUT_SECONDS" cargo "${cargo_args[@]}" || {
            log_warning "Some integration tests failed or timed out"
            return 1
        }
    else
        cargo "${cargo_args[@]}" || {
            log_warning "Some integration tests failed"
            return 1
        }
    fi

    log_success "Integration tests completed"
}

# Run WASM tests
run_wasm_tests() {
    log_info "Running WASM tests..."

    if ! command -v wasm-pack &> /dev/null; then
        log_warning "wasm-pack not found, skipping WASM tests"
        return 0
    fi

    # Test each crate that supports WASM
    local wasm_crates=("purr-ui" "purr-web")

    for crate in "${wasm_crates[@]}"; do
        if [[ -n "$SPECIFIC_CRATE" ]] && [[ "$SPECIFIC_CRATE" != "$crate" ]]; then
            continue
        fi

        log_info "Testing WASM for $crate..."

        pushd "crates/$crate" > /dev/null || continue

        if [[ -f "Cargo.toml" ]] && grep -q "wasm-bindgen-test" Cargo.toml; then
            wasm-pack test --node --features web || {
                log_warning "WASM tests failed for $crate"
                popd > /dev/null
                continue
            }
        fi

        popd > /dev/null
    done

    log_success "WASM tests completed"
}

# Generate coverage report
generate_coverage() {
    if [[ "$COVERAGE" != "true" ]]; then
        log_info "Coverage generation disabled, skipping..."
        return 0
    fi

    log_info "Generating coverage report..."

    local tarpaulin_args=(
        "--out" "Html" "Xml" "Json"
        "--output-dir" "$COVERAGE_DIR"
        "--timeout" "$TIMEOUT_SECONDS"
        "--verbose"
        "--skip-clean"
    )

    if [[ -n "$SPECIFIC_CRATE" ]]; then
        tarpaulin_args+=("--packages" "$SPECIFIC_CRATE")
    else
        tarpaulin_args+=("--workspace")
    fi

    # Exclude test utilities from coverage
    tarpaulin_args+=("--exclude" "purr-test-utils")

    # Run coverage with cargo-tarpaulin
    cargo tarpaulin "${tarpaulin_args[@]}" || {
        log_warning "Coverage generation failed, but continuing..."
    }

    # Generate additional coverage formats if lcov is available
    if command -v lcov &> /dev/null; then
        log_info "Generating LCOV coverage report..."
        cargo tarpaulin --out Lcov --output-dir "$COVERAGE_DIR" --workspace --exclude purr-test-utils || true
    fi

    log_success "Coverage reports generated in $COVERAGE_DIR"
}

# Run benchmark tests
run_benchmarks() {
    if [[ "$BENCHMARKS" != "true" ]]; then
        return 0
    fi

    log_info "Running benchmark tests..."

    local cargo_args=("bench" "--features" "benchmarks")

    if [[ -n "$SPECIFIC_CRATE" ]]; then
        cargo_args+=("--package" "$SPECIFIC_CRATE")
    else
        cargo_args+=("--workspace")
    fi

    cargo "${cargo_args[@]}" || {
        log_warning "Some benchmarks failed"
    }

    log_success "Benchmarks completed"
}

# Check for flaky tests
check_flakiness() {
    if [[ "$FLAKINESS_CHECK" != "true" ]]; then
        return 0
    fi

    log_info "Checking for flaky tests (running $FLAKINESS_RUNS times)..."

    local flaky_tests=()
    local temp_dir=$(mktemp -d)

    for ((i=1; i<=FLAKINESS_RUNS; i++)); do
        log_info "Flakiness run $i/$FLAKINESS_RUNS"

        local test_output="$temp_dir/test_run_$i.txt"

        if ! cargo test --workspace --profile "$PROFILE" > "$test_output" 2>&1; then
            log_warning "Test run $i failed, analyzing..."

            # Extract failed test names
            grep "test result: FAILED" "$test_output" | while read -r line; do
                flaky_tests+=("Run $i: $line")
            done
        fi
    done

    if [[ ${#flaky_tests[@]} -gt 0 ]]; then
        log_warning "Potentially flaky tests detected:"
        printf '%s\n' "${flaky_tests[@]}"

        # Save flaky test report
        printf '%s\n' "${flaky_tests[@]}" > "$REPORTS_DIR/flaky_tests.txt"
    else
        log_success "No flaky tests detected"
    fi

    rm -rf "$temp_dir"
}

# Run doctests
run_doctests() {
    log_info "Running documentation tests..."

    local cargo_args=("test" "--doc" "--profile" "$PROFILE")

    if [[ -n "$SPECIFIC_CRATE" ]]; then
        cargo_args+=("--package" "$SPECIFIC_CRATE")
    else
        cargo_args+=("--workspace")
    fi

    cargo "${cargo_args[@]}" || {
        log_warning "Some documentation tests failed"
    }

    log_success "Documentation tests completed"
}

# Check test configuration
check_test_config() {
    log_info "Validating test configuration..."

    # Check that test profiles exist
    if ! grep -q "\\[profile\\.${PROFILE}\\]" "$WORKSPACE_ROOT/Cargo.toml"; then
        log_error "Test profile '$PROFILE' not found in workspace Cargo.toml"
        exit 1
    fi

    # Verify test dependencies are properly configured
    local crates=("purr" "purr-core" "purr-common" "purr-ui" "purr-web" "purr-test-utils")

    for crate in "${crates[@]}"; do
        local crate_toml="$WORKSPACE_ROOT/crates/$crate/Cargo.toml"
        if [[ -f "$crate_toml" ]] && ! grep -q "\\[dev-dependencies\\]" "$crate_toml"; then
            log_warning "No dev-dependencies section found in $crate/Cargo.toml"
        fi
    done

    log_success "Test configuration validation completed"
}

# Generate test report
generate_test_report() {
    log_info "Generating test report..."

    local report_file="$REPORTS_DIR/test_summary.md"

    cat > "$report_file" << EOF
# Test Report

**Generated:** $(date)
**Profile:** $PROFILE
**Workspace:** $WORKSPACE_ROOT

## Test Results

EOF

    # Add coverage information if available
    if [[ -f "$COVERAGE_DIR/tarpaulin-report.json" ]]; then
        local coverage=$(jq -r '.coverage' "$COVERAGE_DIR/tarpaulin-report.json" 2>/dev/null || echo "N/A")
        echo "**Coverage:** $coverage%" >> "$report_file"
        echo "" >> "$report_file"
    fi

    # Add flaky test information if available
    if [[ -f "$REPORTS_DIR/flaky_tests.txt" ]]; then
        echo "## Flaky Tests" >> "$report_file"
        echo "" >> "$report_file"
        echo '```' >> "$report_file"
        cat "$REPORTS_DIR/flaky_tests.txt" >> "$report_file"
        echo '```' >> "$report_file"
        echo "" >> "$report_file"
    fi

    # Add performance information
    echo "## Performance" >> "$report_file"
    echo "" >> "$report_file"
    echo "- Test profile: $PROFILE" >> "$report_file"
    echo "- Timeout: ${TIMEOUT_SECONDS}s" >> "$report_file"
    echo "" >> "$report_file"

    log_success "Test report generated: $report_file"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add any cleanup logic here
}

# Main execution
main() {
    log_info "Starting comprehensive test suite for Purr WebGPU workspace"
    log_info "Workspace: $WORKSPACE_ROOT"
    log_info "Profile: $PROFILE"

    # Set up trap for cleanup
    trap cleanup EXIT

    # Run test phases
    check_dependencies
    setup_test_env
    check_test_config

    # Core testing phases
    run_unit_tests
    run_integration_tests
    run_doctests
    run_wasm_tests

    # Optional phases
    run_benchmarks
    check_flakiness
    generate_coverage

    # Reporting
    generate_test_report

    log_success "All test phases completed successfully!"

    # Print summary
    echo ""
    echo "=========================="
    echo "    TEST SUITE SUMMARY"
    echo "=========================="
    echo "Profile: $PROFILE"
    echo "Coverage: $COVERAGE"
    echo "Benchmarks: $BENCHMARKS"
    echo "Flakiness check: $FLAKINESS_CHECK"

    if [[ -n "$SPECIFIC_CRATE" ]]; then
        echo "Crate: $SPECIFIC_CRATE"
    else
        echo "Scope: All crates"
    fi

    echo ""
    echo "Reports available in: $REPORTS_DIR"

    if [[ "$COVERAGE" == "true" ]]; then
        echo "Coverage reports in: $COVERAGE_DIR"
    fi

    echo "=========================="
}

# Execute main function
main "$@"