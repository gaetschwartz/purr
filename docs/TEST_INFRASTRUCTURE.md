# Test Infrastructure Documentation

## Overview

This document describes the comprehensive test infrastructure set up for the Purr WebGPU workspace. The infrastructure provides unified testing capabilities across all crates with proper coverage reporting, performance benchmarking, and cross-platform support.

## Components

### 1. Test Utilities Crate (`purr-test-utils`)

A dedicated crate providing shared testing functionality:

- **Location**: `crates/purr-test-utils/`
- **Purpose**: Centralized test utilities, fixtures, mocks, and helpers
- **Features**: Core testing, WebGPU testing, benchmarking, mocking

#### Key Modules:

- `assertions.rs` - Custom assertion macros for Purr-specific patterns
- `fixtures.rs` - Test data generators and sample configurations
- `generators.rs` - Property-based testing data generators
- `mocks.rs` - Mock implementations for external dependencies
- `webgpu.rs` - WebGPU testing utilities and mock GPU implementations
- `benchmarks.rs` - Performance testing and benchmarking utilities

### 2. Test Runner Script

A comprehensive test runner script at `scripts/test-runner.sh`:

#### Features:
- Unit, integration, and documentation tests
- WASM testing support
- Coverage reporting with `cargo-tarpaulin`
- Flakiness detection (multiple test runs)
- Benchmarking with Criterion
- Configurable test profiles
- Timeout handling
- Detailed reporting

#### Usage Examples:
```bash
# Run all tests with coverage
./scripts/test-runner.sh

# Run tests without coverage (faster)
./scripts/test-runner.sh --no-coverage

# Test specific crate
./scripts/test-runner.sh --crate purr-core

# Run benchmarks
./scripts/test-runner.sh --benchmarks

# Check for flaky tests
./scripts/test-runner.sh --flakiness
```

### 3. Cargo Configuration

#### Test Profiles (`Cargo.toml`):
- `test` - Standard test profile with debug information
- `test-opt` - Optimized test profile for performance testing
- `bench` - Benchmark profile with release optimizations

#### Cargo Aliases (`.cargo/config.toml`):
```toml
test-all = "test --workspace --profile test"
test-unit = "test --workspace --lib --profile test"
test-integration = "test --workspace --test '*' --profile test"
coverage = "tarpaulin --workspace --out Html Xml Json ..."
bench-all = "bench --workspace --features benchmarks"
```

### 4. Coverage Configuration

#### Tarpaulin Configuration (`tarpaulin.toml`):
- HTML, XML, JSON, and LCOV output formats
- Workspace-wide coverage
- Appropriate exclusions (test utilities, benches)
- 70% coverage threshold
- LLVM engine for accurate coverage

### 5. Build Automation (`justfile`)

Convenient commands for development workflows:

```bash
just test          # Run all tests with coverage
just test-fast     # Run tests without coverage
just bench         # Run benchmarks
just coverage-open # Generate and open coverage report
just lint          # Run clippy linting
just fmt           # Format code
just check-all     # Run all quality checks
```

## Test Organization

### Dependency Management

Each crate includes appropriate test dependencies:

- **Core Testing**: `rstest`, `pretty_assertions`, `tokio-test`, `tempfile`
- **Property Testing**: `proptest`, `proptest-derive`
- **Mocking**: `mockall`
- **Benchmarking**: `criterion` (feature-gated)
- **WASM Testing**: `wasm-bindgen-test` (for web crates)

### Feature Gates

Test-related features are properly gated:
- `benchmarks` - Enables benchmark dependencies and tests
- `mocks` - Enables mocking capabilities
- `web` - Enables WebGPU and WASM testing utilities

## Test Types

### 1. Unit Tests
- Standard `#[test]` functions
- Use `rstest` for parameterized tests
- Mock external dependencies

### 2. Integration Tests
- Located in `tests/` directories
- Test crate-level functionality
- Use real implementations where appropriate

### 3. Property-Based Tests
- Use `proptest` for generating test data
- Custom generators in `purr-test-utils`
- Validate invariants across input ranges

### 4. Benchmark Tests
- Use `criterion` for performance measurements
- Audio processing benchmarks
- WebGPU operation benchmarks
- Memory usage analysis

### 5. WASM Tests
- Browser-based testing with `wasm-bindgen-test`
- WebGPU functionality testing
- Cross-platform compatibility

## Assertion Macros

Custom assertion macros for domain-specific testing:

```rust
// Async operation timeouts
assert_async_timeout!(100, async_operation());

// WebGPU operations
assert_webgpu_ok!(gpu_result);

// Audio data validation
assert_audio_valid!(audio_data, sample_rate, channels);

// Transcription result validation
assert_transcription_valid!(transcription);

// File size validation
assert_file_size_range!(path, min_bytes, max_bytes);

// Memory usage checks
assert_memory_usage!(max_mb, || { operation(); });
```

## Mock Implementations

Comprehensive mocking for external dependencies:

- **Audio Processing**: Mock audio processor for testing without actual audio
- **Transcription Engine**: Mock inference without loading models
- **File System**: Mock file operations for isolated testing
- **Network Client**: Mock downloads and API calls
- **WebGPU**: Mock GPU operations for testing without hardware
- **Configuration**: Mock configuration management

## Performance Testing

### Benchmarking Suite
- Audio processing throughput
- WebGPU compute operations
- Memory allocation patterns
- Async operation overhead

### Simple Profiler
- Time synchronous and asynchronous operations
- Collect statistics (mean, min, max, std dev)
- Memory usage tracking
- Cross-platform compatibility

## WebGPU Testing

### Test Environment
- Mock WebGPU implementations
- Browser-compatible testing
- Compute shader validation
- Performance benchmarking

### Mock GPU
- Simulated GPU operations
- Configurable operations
- CPU-based validation
- Testing without hardware requirements

## Configuration Files

### Key Configuration Files:
- `Cargo.toml` - Workspace and crate dependencies
- `.cargo/config.toml` - Build configuration and aliases
- `tarpaulin.toml` - Coverage configuration
- `justfile` - Build automation
- `scripts/test-runner.sh` - Comprehensive test execution

## Environment Variables

Test environment configuration:
- `RUST_BACKTRACE=1` - Enable backtraces
- `RUST_LOG=debug` - Enable debug logging
- `WGPU_BACKEND` - WebGPU backend selection
- `WASM_BINDGEN_TEST_TIMEOUT` - WASM test timeout

## Reporting

### Coverage Reports
- HTML reports with line-by-line coverage
- XML reports for CI integration
- JSON reports for programmatic analysis
- LCOV reports for editor integration

### Test Reports
- Comprehensive test summaries
- Performance metrics
- Flaky test detection
- Failed test analysis

## Platform Support

### Supported Platforms:
- **Native**: Linux, macOS, Windows
- **WASM**: Browser-based testing
- **Cross-compilation**: Various target architectures

### Platform-Specific Features:
- Memory profiling (Linux with valgrind)
- GPU backend selection
- Platform-specific test exclusions

## Best Practices

### Test Organization:
1. Use descriptive test names
2. Group related tests with `rstest`
3. Separate unit and integration tests
4. Mock external dependencies
5. Use property-based testing for invariants

### Performance Testing:
1. Use appropriate benchmark profiles
2. Isolate performance-critical paths
3. Test with realistic data sizes
4. Monitor memory usage
5. Validate across platforms

### WebGPU Testing:
1. Test with mock implementations first
2. Validate shader compilation
3. Test error handling paths
4. Benchmark compute operations
5. Ensure cross-browser compatibility

## Continuous Integration

The test infrastructure is designed for CI environments:

- Timeout handling for hung tests
- Parallel test execution
- Coverage threshold enforcement
- Platform-specific test selection
- Artifact generation for debugging

## Future Enhancements

Potential improvements:
- Integration with fuzzing tools
- Mutation testing
- Performance regression detection
- Automated benchmark comparisons
- Enhanced WASM testing capabilities

## Troubleshooting

### Common Issues:

1. **Test Timeouts**: Increase timeout values or optimize test performance
2. **WASM Test Failures**: Check browser compatibility and WebGPU support
3. **Coverage Issues**: Verify tarpaulin configuration and exclusions
4. **Mock Failures**: Ensure proper mock setup and expectations
5. **Platform Differences**: Use platform-specific test attributes

### Debug Commands:
```bash
# Check workspace configuration
cargo check --workspace

# Run specific test with verbose output
cargo test test_name -- --nocapture

# Generate coverage with debug info
cargo tarpaulin --verbose --debug

# Run benchmarks with specific features
cargo bench --features benchmarks
```

This test infrastructure provides a solid foundation for maintaining code quality, performance, and reliability across the entire Purr WebGPU workspace.