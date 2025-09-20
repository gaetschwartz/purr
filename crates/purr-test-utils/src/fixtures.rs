//! Test fixtures and sample data for Purr testing

use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::TempDir;

/// Test audio file fixtures
pub struct AudioFixtures {
    pub temp_dir: TempDir,
}

impl AudioFixtures {
    pub fn new() -> Self {
        Self {
            temp_dir: tempfile::tempdir().expect("Failed to create temp dir"),
        }
    }

    /// Generate a simple sine wave as test audio data
    pub fn generate_sine_wave(
        &self,
        duration_secs: f32,
        frequency: f32,
        sample_rate: u32,
    ) -> Vec<f32> {
        let num_samples = (duration_secs * sample_rate as f32) as usize;
        let mut samples = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
            samples.push(sample);
        }

        samples
    }

    /// Generate white noise for testing
    pub fn generate_white_noise(&self, duration_secs: f32, sample_rate: u32) -> Vec<f32> {
        use fake::{Fake, Faker};

        let num_samples = (duration_secs * sample_rate as f32) as usize;
        (0..num_samples)
            .map(|_| Faker.fake::<f32>() * 2.0 - 1.0) // Range [-1.0, 1.0]
            .collect()
    }

    /// Create a temporary WAV file with test audio
    pub fn create_wav_file(&self, audio_data: &[f32], sample_rate: u32) -> PathBuf {
        let file_path = self.temp_dir.path().join("test_audio.wav");

        // Simple WAV header creation (simplified for testing)
        let mut wav_data = Vec::new();

        // WAV header
        wav_data.extend_from_slice(b"RIFF");
        wav_data.extend_from_slice(&(36 + audio_data.len() * 4).to_le_bytes());
        wav_data.extend_from_slice(b"WAVE");

        // Format chunk
        wav_data.extend_from_slice(b"fmt ");
        wav_data.extend_from_slice(&16u32.to_le_bytes()); // Chunk size
        wav_data.extend_from_slice(&1u16.to_le_bytes());  // Audio format (PCM)
        wav_data.extend_from_slice(&1u16.to_le_bytes());  // Channels
        wav_data.extend_from_slice(&sample_rate.to_le_bytes());
        wav_data.extend_from_slice(&(sample_rate * 4).to_le_bytes()); // Byte rate
        wav_data.extend_from_slice(&4u16.to_le_bytes());  // Block align
        wav_data.extend_from_slice(&32u16.to_le_bytes()); // Bits per sample

        // Data chunk
        wav_data.extend_from_slice(b"data");
        wav_data.extend_from_slice(&(audio_data.len() * 4).to_le_bytes());

        // Audio data (convert f32 to i32 for WAV)
        for &sample in audio_data {
            let int_sample = (sample * i32::MAX as f32) as i32;
            wav_data.extend_from_slice(&int_sample.to_le_bytes());
        }

        std::fs::write(&file_path, wav_data).expect("Failed to write WAV file");
        file_path
    }
}

impl Default for AudioFixtures {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration fixtures for testing
pub struct ConfigFixtures;

impl ConfigFixtures {
    /// Create a minimal valid configuration
    pub fn minimal_config() -> Value {
        serde_json::json!({
            "model": {
                "name": "base",
                "language": "auto"
            },
            "audio": {
                "sample_rate": 16000,
                "channels": 1
            },
            "output": {
                "format": "json"
            }
        })
    }

    /// Create a full configuration with all options
    pub fn full_config() -> Value {
        serde_json::json!({
            "model": {
                "name": "large-v3",
                "language": "en",
                "translate": false
            },
            "audio": {
                "sample_rate": 16000,
                "channels": 1,
                "normalize": true,
                "denoise": false
            },
            "transcription": {
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0.0,
                "compression_ratio_threshold": 2.4,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": true
            },
            "output": {
                "format": "json",
                "timestamps": true,
                "word_timestamps": false
            },
            "performance": {
                "threads": 4,
                "gpu_enabled": true
            }
        })
    }

    /// Create an invalid configuration for error testing
    pub fn invalid_config() -> Value {
        serde_json::json!({
            "model": {
                "name": "", // Invalid empty name
                "language": "invalid_lang"
            },
            "audio": {
                "sample_rate": 0, // Invalid sample rate
                "channels": -1    // Invalid channel count
            }
        })
    }
}

/// WebGPU test fixtures
#[cfg(feature = "web")]
pub struct WebGpuFixtures;

#[cfg(feature = "web")]
impl WebGpuFixtures {
    /// Create mock WebGPU adapter for testing
    pub fn mock_adapter() -> MockGpuAdapter {
        MockGpuAdapter::new()
    }

    /// Create mock WebGPU device for testing
    pub fn mock_device() -> MockGpuDevice {
        MockGpuDevice::new()
    }

    /// Generate test compute shader source
    pub fn test_compute_shader() -> &'static str {
        r#"
        @group(0) @binding(0)
        var<storage, read_write> data: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= arrayLength(&data)) {
                return;
            }
            data[index] = data[index] * 2.0;
        }
        "#
    }

    /// Generate test data for WebGPU operations
    pub fn test_buffer_data(size: usize) -> Vec<f32> {
        (0..size).map(|i| i as f32).collect()
    }
}

/// Network fixtures for testing web requests
pub struct NetworkFixtures;

impl NetworkFixtures {
    /// Create a mock HTTP response for model downloads
    pub fn mock_model_download_response() -> HashMap<String, Value> {
        let mut response = HashMap::new();
        response.insert(
            "headers".to_string(),
            serde_json::json!({
                "content-length": "1048576",
                "content-type": "application/octet-stream"
            }),
        );
        response.insert(
            "status".to_string(),
            serde_json::json!(200),
        );
        response
    }

    /// Create mock error responses
    pub fn mock_error_response(status: u16, message: &str) -> HashMap<String, Value> {
        let mut response = HashMap::new();
        response.insert(
            "status".to_string(),
            serde_json::json!(status),
        );
        response.insert(
            "error".to_string(),
            serde_json::json!({
                "message": message,
                "code": status
            }),
        );
        response
    }
}

/// Performance test fixtures
pub struct PerformanceFixtures;

impl PerformanceFixtures {
    /// Generate large audio data for performance testing
    pub fn large_audio_data(minutes: u32) -> Vec<f32> {
        let sample_rate = 16000;
        let samples_per_minute = sample_rate * 60;
        let _total_samples = samples_per_minute * minutes;

        AudioFixtures::new().generate_sine_wave(
            (minutes * 60) as f32,
            440.0, // A4 note
            sample_rate,
        )
    }

    /// Create stress test scenarios
    pub fn stress_test_scenarios() -> Vec<(String, usize)> {
        vec![
            ("Small file".to_string(), 1024),
            ("Medium file".to_string(), 1024 * 1024),
            ("Large file".to_string(), 10 * 1024 * 1024),
            ("Extra large file".to_string(), 100 * 1024 * 1024),
        ]
    }
}

#[cfg(feature = "web")]
pub struct MockGpuAdapter;

#[cfg(feature = "web")]
impl MockGpuAdapter {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "web")]
pub struct MockGpuDevice;

#[cfg(feature = "web")]
impl MockGpuDevice {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_fixtures_sine_wave() {
        let fixtures = AudioFixtures::new();
        let audio = fixtures.generate_sine_wave(1.0, 440.0, 16000);

        assert_eq!(audio.len(), 16000);
        assert!(audio.iter().all(|&sample| sample >= -1.0 && sample <= 1.0));
    }

    #[test]
    fn test_config_fixtures() {
        let config = ConfigFixtures::minimal_config();
        assert!(config["model"]["name"].is_string());
        assert!(config["audio"]["sample_rate"].is_number());
    }

    #[test]
    fn test_performance_fixtures() {
        let scenarios = PerformanceFixtures::stress_test_scenarios();
        assert!(!scenarios.is_empty());
        assert!(scenarios.iter().all(|(name, size)| !name.is_empty() && *size > 0));
    }
}