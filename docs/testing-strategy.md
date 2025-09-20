# Testing Strategy for Purr WebGPU Project

## Executive Summary

After analyzing the purr-webgpu project structure and existing test coverage, I've identified critical gaps in testing, particularly for WebGPU-specific functionality. This document outlines a comprehensive testing strategy to ensure robust coverage of all components.

## Current Test Coverage Analysis

### Existing Tests ✅
- **purr-core**: Basic integration tests, configuration tests, transcription workflow tests
- **purr-web**: Audio format validation tests only
- **purr**: CLI integration tests with sample files
- **purr-ui**: No tests found
- **purr-common**: No tests found

### Critical Gaps Identified ⚠️

#### High Priority - Missing Tests
1. **WebGPU Core Functionality** (0% coverage)
   - `purr-web/src/worker.rs` (427 lines) - WebGPU transcription worker
   - `purr-web/src/platform.rs` (500 lines) - Web platform integration
   - `purr-web/src/indexeddb.rs` (100+ lines) - Browser storage

2. **Audio Processing Core** (0% coverage)
   - `purr-core/src/audio.rs` (100+ lines) - FFmpeg audio processing
   - `purr-core/src/model.rs` (100+ lines) - Model management

3. **Platform Abstraction** (0% coverage)
   - `purr-common/src/platform.rs` (606 lines) - Platform trait definitions

4. **Math & Performance** (Partial coverage)
   - `purr-core/src/math.rs` (290 lines) - Has basic tests but needs expansion

## Testing Strategy by Component

### 1. WebGPU Worker Testing (`purr-web/src/worker.rs`)

#### Unit Tests Needed:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Configuration tests
    #[test]
    fn test_transcription_config_defaults() { /* ... */ }

    #[test]
    fn test_transcription_config_validation() { /* ... */ }

    // Worker message serialization
    #[test]
    fn test_worker_message_serialization() { /* ... */ }

    #[test]
    fn test_worker_response_deserialization() { /* ... */ }

    // Session management
    #[tokio::test]
    async fn test_session_creation() { /* ... */ }

    #[tokio::test]
    async fn test_session_cleanup() { /* ... */ }

    // Progress status conversion
    #[test]
    fn test_progress_status_conversion() { /* ... */ }
}
```

#### Mock Framework Requirements:
- Mock WebWorker implementation
- Mock MessageEvent handling
- Mock WebGPU APIs for testing without actual GPU

### 2. IndexedDB Storage Testing (`purr-web/src/indexeddb.rs`)

#### Unit Tests Needed:
```rust
#[cfg(test)]
mod tests {
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_database_initialization() { /* ... */ }

    #[wasm_bindgen_test]
    async fn test_model_storage_and_retrieval() { /* ... */ }

    #[wasm_bindgen_test]
    async fn test_download_progress_tracking() { /* ... */ }

    #[wasm_bindgen_test]
    async fn test_metadata_operations() { /* ... */ }

    #[wasm_bindgen_test]
    async fn test_storage_quota_handling() { /* ... */ }
}
```

#### Browser Test Setup:
- Use `wasm-bindgen-test` for browser-specific tests
- Mock IndexedDB API for unit tests
- Test actual browser storage in integration tests

### 3. Audio Processing Testing (`purr-core/src/audio.rs`)

#### Unit Tests Needed:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_audio_processor_creation() { /* ... */ }

    #[tokio::test]
    async fn test_audio_loading_wav() { /* ... */ }

    #[tokio::test]
    async fn test_audio_loading_mp3() { /* ... */ }

    #[tokio::test]
    async fn test_audio_streaming() { /* ... */ }

    #[tokio::test]
    async fn test_chunk_processing() { /* ... */ }

    #[test]
    fn test_audio_chunk_creation() { /* ... */ }

    #[tokio::test]
    async fn test_error_handling_invalid_files() { /* ... */ }
}
```

#### Test Fixtures Required:
```
tests/fixtures/audio/
├── test_samples/
│   ├── test_mono_16khz.wav      # 1 second mono audio
│   ├── test_stereo_44khz.wav    # 2 seconds stereo audio
│   ├── test_compressed.mp3      # 3 seconds MP3
│   ├── test_invalid.txt         # Invalid audio file
│   └── test_corrupted.wav       # Corrupted WAV header
└── expected_outputs/
    ├── mono_processed.json      # Expected AudioData
    └── chunk_sequence.json      # Expected chunk sequence
```

### 4. Platform Abstraction Testing (`purr-common/src/platform.rs`)

#### Unit Tests Needed:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Error type tests
    #[test]
    fn test_platform_error_creation() { /* ... */ }

    #[test]
    fn test_platform_error_conversion() { /* ... */ }

    // FileId tests
    #[test]
    fn test_file_id_creation_and_conversion() { /* ... */ }

    // ModelInfo tests
    #[test]
    fn test_model_info_serialization() { /* ... */ }

    #[test]
    fn test_model_metadata_builder_pattern() { /* ... */ }

    // Status enum tests
    #[test]
    fn test_transcription_status_serialization() { /* ... */ }

    #[test]
    fn test_processing_status_serialization() { /* ... */ }
}
```

### 5. Enhanced Math Testing (`purr-core/src/math.rs`)

#### Additional Tests Needed:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Expand existing tests
    #[test]
    fn test_byte_speed_edge_cases() { /* ... */ }

    #[test]
    fn test_duration_range_operations() { /* ... */ }

    #[test]
    fn test_performance_calculations() { /* ... */ }

    #[test]
    fn test_memory_size_formatting() { /* ... */ }

    // New performance tests
    #[test]
    fn test_large_number_calculations() { /* ... */ }

    #[test]
    fn test_precision_edge_cases() { /* ... */ }
}
```

## WebGPU-Specific Testing Considerations

### 1. Mock WebGPU Environment
```rust
// Mock traits for testing
pub trait MockWebGpu {
    fn create_device(&self) -> MockGpuDevice;
    fn create_buffer(&self, size: usize) -> MockGpuBuffer;
    fn submit_commands(&self, commands: Vec<MockCommand>);
}

// Test utilities
pub struct WebGpuTestUtils {
    mock_gpu: MockWebGpu,
}

impl WebGpuTestUtils {
    pub fn new() -> Self { /* ... */ }
    pub fn simulate_transcription(&self, audio_data: &[u8]) -> Result<TranscriptionResult> { /* ... */ }
    pub fn verify_gpu_memory_usage(&self) -> GpuMemoryStats { /* ... */ }
}
```

### 2. Browser Integration Tests
```rust
// Browser-specific tests using wasm-bindgen-test
#[wasm_bindgen_test]
async fn test_webgpu_device_detection() {
    // Test WebGPU availability
    // Test device capabilities
    // Test adapter selection
}

#[wasm_bindgen_test]
async fn test_webgpu_memory_limits() {
    // Test memory allocation limits
    // Test buffer size constraints
    // Test concurrent operation limits
}
```

### 3. Performance Benchmarks
```rust
#[cfg(test)]
mod bench_tests {
    use criterion::{criterion_group, criterion_main, Criterion};

    fn bench_webgpu_transcription(c: &mut Criterion) {
        c.bench_function("webgpu_transcription_1min", |b| {
            b.iter(|| {
                // Benchmark 1-minute audio transcription
            });
        });
    }

    criterion_group!(benches, bench_webgpu_transcription);
    criterion_main!(benches);
}
```

## Test Infrastructure Requirements

### 1. Additional Dependencies
```toml
[dev-dependencies]
# Existing
tempfile = "3.21"
rstest = "0.26"
pretty_assertions = "1.4.1"

# New additions needed
wasm-bindgen-test = "0.3"
criterion = "0.5"
mockall = "0.12"
tokio-test = "0.4"
fake = "2.9"
proptest = "1.4"
```

### 2. CI/CD Integration
```yaml
# .github/workflows/test.yml additions
- name: Test WebGPU components
  run: |
    # Run WASM tests in browser environment
    wasm-pack test --headless --firefox crates/purr-web
    wasm-pack test --headless --chrome crates/purr-web

- name: Run performance benchmarks
  run: |
    cargo bench --workspace
```

### 3. Test Organization
```
tests/
├── unit/                    # Unit tests per module
│   ├── webgpu/
│   ├── audio/
│   └── platform/
├── integration/             # Cross-component tests
│   ├── webgpu_platform/
│   └── audio_transcription/
├── browser/                 # Browser-specific tests
│   ├── indexeddb/
│   └── webworker/
├── fixtures/                # Test data
│   ├── audio_samples/
│   ├── model_metadata/
│   └── mock_responses/
└── benchmarks/              # Performance tests
    ├── transcription/
    └── memory_usage/
```

## Implementation Priority

### Phase 1: Critical Foundation (Week 1)
1. **WebGPU Worker Tests** - Core functionality testing
2. **Audio Processing Tests** - FFmpeg integration testing
3. **Platform Abstraction Tests** - Error handling and serialization

### Phase 2: Storage & Integration (Week 2)
1. **IndexedDB Storage Tests** - Browser storage testing
2. **Integration Tests** - Cross-component workflows
3. **Mock Framework Setup** - WebGPU mocking infrastructure

### Phase 3: Performance & Edge Cases (Week 3)
1. **Performance Benchmarks** - Memory and speed testing
2. **Edge Case Testing** - Error conditions and limits
3. **Browser Compatibility Tests** - Cross-browser validation

## Success Metrics

### Coverage Targets
- **Unit Test Coverage**: 85%+ for all core modules
- **Integration Test Coverage**: 70%+ for cross-component workflows
- **WebGPU-Specific Coverage**: 90%+ for WebGPU-related code

### Performance Benchmarks
- **Transcription Speed**: <2x real-time for base model
- **Memory Usage**: <500MB peak for typical workflows
- **Startup Time**: <3 seconds for WebGPU initialization

### Quality Gates
- All tests must pass in CI/CD
- No memory leaks in browser tests
- Cross-browser compatibility (Chrome, Firefox, Safari)
- Performance regression detection

## Conclusion

This testing strategy addresses the critical gaps in the purr-webgpu project, with particular focus on WebGPU functionality that currently lacks any test coverage. The phased implementation approach ensures that the most critical components are tested first, while the comprehensive fixture and mock framework provides a solid foundation for ongoing development.

The emphasis on WebGPU-specific testing considerations reflects the unique challenges of testing browser-based GPU acceleration, including mock frameworks, browser integration tests, and performance benchmarks that are essential for this type of application.