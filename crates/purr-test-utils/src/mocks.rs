//! Mock implementations for external dependencies and services

use mockall::mock;
use std::path::Path;
use std::time::Duration;
use tokio::sync::mpsc;
use bytes::Bytes;

// Mock audio processor for testing transcription without actual audio processing
mock! {
    pub AudioProcessor {
        async fn process_audio(&self, audio_data: &[f32], sample_rate: u32) -> Result<Vec<f32>, String>;
        async fn normalize_audio(&self, audio_data: &[f32]) -> Vec<f32>;
        async fn resample_audio(&self, audio_data: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>, String>;
        fn supported_formats(&self) -> Vec<String>;
    }
}

// Mock transcription engine for testing without actual model inference
mock! {
    pub TranscriptionEngine {
        async fn transcribe(&self, audio_data: &[f32], options: TranscriptionOptions) -> Result<TranscriptionResult, String>;
        async fn load_model(&mut self, model_path: &Path) -> Result<(), String>;
        fn is_model_loaded(&self) -> bool;
        fn supported_languages(&self) -> Vec<String>;
    }
}

// Mock file system operations for testing without actual file I/O
mock! {
    pub FileSystem {
        async fn read_file(&self, path: &Path) -> Result<Bytes, std::io::Error>;
        async fn write_file(&self, path: &Path, data: &[u8]) -> Result<(), std::io::Error>;
        async fn exists(&self, path: &Path) -> bool;
        async fn create_dir_all(&self, path: &Path) -> Result<(), std::io::Error>;
        async fn remove_file(&self, path: &Path) -> Result<(), std::io::Error>;
        async fn file_size(&self, path: &Path) -> Result<u64, std::io::Error>;
    }
}

// Mock network client for testing downloads and API calls
mock! {
    pub NetworkClient {
        async fn download_file(&self, url: &str, destination: &Path) -> Result<(), NetworkError>;
        async fn download_with_progress(&self, url: &str, destination: &Path, progress_tx: mpsc::Sender<DownloadProgress>) -> Result<(), NetworkError>;
        async fn get_file_size(&self, url: &str) -> Result<u64, NetworkError>;
        async fn head_request(&self, url: &str) -> Result<ResponseHeaders, NetworkError>;
    }
}

/// Mock WebGPU device for testing GPU operations without actual hardware
#[cfg(target_arch = "wasm32")]
mock! {
    pub WebGpuDevice {
        async fn create_buffer(&self, size: u64, usage: u32) -> Result<MockBuffer, WebGpuError>;
        async fn create_compute_pipeline(&self, shader_source: &str) -> Result<MockComputePipeline, WebGpuError>;
        async fn create_bind_group(&self, buffers: &[&MockBuffer]) -> Result<MockBindGroup, WebGpuError>;
        async fn submit_commands(&self, commands: Vec<MockCommand>) -> Result<(), WebGpuError>;
        fn queue(&self) -> &MockQueue;
    }
}

#[cfg(target_arch = "wasm32")]
mock! {
    pub Buffer {
        fn size(&self) -> u64;
        fn usage(&self) -> u32;
        async fn map_read(&self) -> Result<&[u8], WebGpuError>;
        async fn unmap(&self);
    }
}

#[cfg(target_arch = "wasm32")]
mock! {
    pub ComputePipeline {
        fn workgroup_size(&self) -> (u32, u32, u32);
    }
}

#[cfg(target_arch = "wasm32")]
mock! {
    pub BindGroup {
        fn id(&self) -> u32;
    }
}

#[cfg(target_arch = "wasm32")]
mock! {
    pub Queue {
        fn write_buffer(&self, buffer: &MockBuffer, offset: u64, data: &[u8]);
        fn submit(&self, commands: &[MockCommand]);
    }
}

#[cfg(target_arch = "wasm32")]
mock! {
    pub Command {
        fn dispatch(&self, x: u32, y: u32, z: u32);
    }
}

// Mock configuration manager for testing configuration scenarios
mock! {
    pub ConfigManager {
        fn load_config(&self, path: &Path) -> Result<Configuration, ConfigError>;
        fn save_config(&self, config: &Configuration, path: &Path) -> Result<(), ConfigError>;
        fn validate_config(&self, config: &Configuration) -> Result<(), Vec<ConfigError>>;
        fn get_default_config(&self) -> Configuration;
    }
}

// Mock progress reporter for testing progress tracking
mock! {
    pub ProgressReporter {
        fn report_progress(&self, current: u64, total: u64, message: String);
        fn start_task(&self, task_name: String, total_steps: u64);
        fn complete_task(&self, task_name: String, success: bool);
        fn report_error(&self, error: String);
    }
}

// Mock metrics collector for testing performance monitoring
mock! {
    pub MetricsCollector {
        fn start_timer(&self, name: String) -> TimerHandle;
        fn record_counter(&self, name: String, value: u64);
        fn record_gauge(&self, name: String, value: f64);
        fn record_histogram(&self, name: String, value: f64);
        fn get_metrics(&self) -> MetricsSnapshot;
    }
}

// Supporting types for mocks

#[derive(Debug, Clone)]
pub struct TranscriptionOptions {
    pub language: String,
    pub model: String,
    pub beam_size: usize,
    pub temperature: f32,
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub confidence: f32,
    pub duration: Duration,
    pub segments: Vec<TranscriptionSegment>,
}

#[derive(Debug, Clone)]
pub struct TranscriptionSegment {
    pub start: Duration,
    pub end: Duration,
    pub text: String,
    pub confidence: f32,
}

#[derive(Debug)]
pub enum NetworkError {
    ConnectionFailed,
    Timeout,
    InvalidUrl,
    ServerError(u16),
    IoError(String),
}

#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub downloaded: u64,
    pub total: Option<u64>,
    pub speed: f64, // bytes per second
}

#[derive(Debug, Clone)]
pub struct ResponseHeaders {
    pub content_length: Option<u64>,
    pub content_type: Option<String>,
    pub last_modified: Option<String>,
}

#[cfg(target_arch = "wasm32")]
#[derive(Debug)]
pub enum WebGpuError {
    DeviceLost,
    ValidationError(String),
    OutOfMemory,
    Internal(String),
}

#[derive(Debug, Clone)]
pub struct Configuration {
    pub model_name: String,
    pub language: String,
    pub sample_rate: u32,
    pub batch_size: usize,
}

#[derive(Debug)]
pub enum ConfigError {
    InvalidPath,
    ParseError(String),
    ValidationError(String),
    IoError(String),
}

pub struct TimerHandle {
    pub name: String,
    pub start_time: std::time::Instant,
}

impl TimerHandle {
    pub fn stop(self) -> Duration {
        self.start_time.elapsed()
    }
}

#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub counters: std::collections::HashMap<String, u64>,
    pub gauges: std::collections::HashMap<String, f64>,
    pub histograms: std::collections::HashMap<String, Vec<f64>>,
    pub timers: std::collections::HashMap<String, Duration>,
}

/// Factory functions for creating commonly used mocks
pub fn create_mock_audio_processor() -> MockAudioProcessor {
    let mut mock = MockAudioProcessor::new();

    // Set up default expectations
    mock.expect_process_audio()
        .returning(|audio_data, _sample_rate| {
            Ok(audio_data.to_vec()) // Identity transform by default
        });

    mock.expect_normalize_audio()
        .returning(|audio_data| {
            // Simple normalization mock
            let max_amplitude = audio_data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
            if max_amplitude > 0.0 {
                audio_data.iter().map(|&x| x / max_amplitude).collect()
            } else {
                audio_data.to_vec()
            }
        });

    mock.expect_supported_formats()
        .returning(|| vec!["wav".to_string(), "mp3".to_string(), "flac".to_string()]);

    mock
}

pub fn create_mock_transcription_engine() -> MockTranscriptionEngine {
    let mut mock = MockTranscriptionEngine::new();

    mock.expect_transcribe()
        .returning(|_audio_data, _options| {
            Ok(TranscriptionResult {
                text: "Mock transcription result".to_string(),
                confidence: 0.95,
                duration: Duration::from_secs(1),
                segments: vec![TranscriptionSegment {
                    start: Duration::ZERO,
                    end: Duration::from_secs(1),
                    text: "Mock transcription result".to_string(),
                    confidence: 0.95,
                }],
            })
        });

    mock.expect_is_model_loaded()
        .returning(|| true);

    mock.expect_supported_languages()
        .returning(|| vec!["en".to_string(), "es".to_string(), "fr".to_string()]);

    mock
}

pub fn create_mock_network_client() -> MockNetworkClient {
    let mut mock = MockNetworkClient::new();

    mock.expect_get_file_size()
        .returning(|_url| Ok(1024 * 1024)); // 1MB default

    mock.expect_head_request()
        .returning(|_url| {
            Ok(ResponseHeaders {
                content_length: Some(1024 * 1024),
                content_type: Some("application/octet-stream".to_string()),
                last_modified: Some("Wed, 21 Oct 2015 07:28:00 GMT".to_string()),
            })
        });

    mock
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_audio_processor() {
        let processor = create_mock_audio_processor();
        let audio_data = vec![0.5, -0.3, 0.8, -0.1];

        let result = processor.process_audio(&audio_data, 16000).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), audio_data);
    }

    #[tokio::test]
    async fn test_mock_transcription_engine() {
        let engine = create_mock_transcription_engine();
        let audio_data = vec![0.1, 0.2, 0.3];
        let options = TranscriptionOptions {
            language: "en".to_string(),
            model: "base".to_string(),
            beam_size: 5,
            temperature: 0.0,
        };

        let result = engine.transcribe(&audio_data, options).await;
        assert!(result.is_ok());

        let transcription = result.unwrap();
        assert!(!transcription.text.is_empty());
        assert!(transcription.confidence > 0.0);
    }

    #[test]
    fn test_mock_network_client() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let client = create_mock_network_client();
            let size = client.get_file_size("https://example.com/model.bin").await;
            assert!(size.is_ok());
            assert_eq!(size.unwrap(), 1024 * 1024);
        });
    }
}