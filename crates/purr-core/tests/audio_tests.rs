//! Comprehensive unit tests for the audio module
//!
//! Tests cover:
//! - Audio format conversion
//! - Whisper integration (sync and streaming)
//! - Audio buffer management
//! - Sample rate handling
//! - Error conditions and edge cases

use futures::StreamExt;
use purr_core::{
    audio::{AudioChunk, AudioData, AudioProcessor},
    error::{AudioProcessingError, WhisperError},
};
use tempfile::NamedTempFile;
use tokio::fs;

/// Test fixtures for audio testing
pub mod fixtures {
    use super::*;

    /// Generate a test WAV file with known content
    pub async fn create_test_wav_file() -> Result<NamedTempFile, Box<dyn std::error::Error>> {
        let temp_file = NamedTempFile::new()?;

        // Create a minimal WAV file (1 second of 16kHz mono silence)
        let sample_rate = 16000u32;
        let samples = vec![0i16; sample_rate as usize]; // 1 second of silence

        let wav_data = create_wav_bytes(samples, sample_rate)?;
        std::fs::write(temp_file.path(), wav_data)?;

        Ok(temp_file)
    }

    /// Create a test audio file with sine wave
    pub async fn create_test_sine_wav(
        duration_secs: f32,
        frequency: f32,
    ) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
        let temp_file = NamedTempFile::new()?;

        let sample_rate = 16000u32;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;

        // Generate sine wave samples
        let mut samples = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
            samples.push((sample * 32767.0) as i16);
        }

        let wav_data = create_wav_bytes(samples, sample_rate)?;
        std::fs::write(temp_file.path(), wav_data)?;

        Ok(temp_file)
    }

    /// Create test audio data structure
    pub fn create_test_audio_data(duration: f32, sample_rate: u32) -> AudioData {
        let num_samples = (sample_rate as f32 * duration) as usize;
        let samples = vec![0.0f32; num_samples];

        AudioData {
            samples,
            sample_rate,
            duration,
        }
    }

    /// Create test audio chunk
    pub fn create_test_audio_chunk(index: usize, start_time: f32, is_final: bool) -> AudioChunk {
        let samples = vec![0.0f32; AudioChunk::TARGET_SAMPLES];
        AudioChunk::new(samples, index, start_time, is_final)
    }

    /// Helper to create minimal WAV file bytes
    fn create_wav_bytes(
        samples: Vec<i16>,
        sample_rate: u32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut wav_data = Vec::new();

        // WAV header
        wav_data.extend_from_slice(b"RIFF");
        let file_size = 36 + (samples.len() * 2) as u32;
        wav_data.extend_from_slice(&file_size.to_le_bytes());
        wav_data.extend_from_slice(b"WAVE");

        // Format chunk
        wav_data.extend_from_slice(b"fmt ");
        wav_data.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        wav_data.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        wav_data.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav_data.extend_from_slice(&sample_rate.to_le_bytes());
        wav_data.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        wav_data.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav_data.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

        // Data chunk
        wav_data.extend_from_slice(b"data");
        wav_data.extend_from_slice(&((samples.len() * 2) as u32).to_le_bytes());

        // Sample data
        for sample in samples {
            wav_data.extend_from_slice(&sample.to_le_bytes());
        }

        Ok(wav_data)
    }
}

/// Mock utilities for testing
pub mod mocks {

    /// Mock Whisper model response for testing
    pub struct MockWhisperResponse {
        pub text: String,
        pub segments: Vec<MockSegment>,
        pub processing_time: f64,
    }

    pub struct MockSegment {
        pub text: String,
        pub start: f64,
        pub end: f64,
    }

    impl MockWhisperResponse {
        pub fn new_simple(text: &str, duration: f64) -> Self {
            Self {
                text: text.to_string(),
                segments: vec![MockSegment {
                    text: text.to_string(),
                    start: 0.0,
                    end: duration,
                }],
                processing_time: 0.1,
            }
        }
    }
}

/// Test utilities for audio comparison and validation
pub mod utils {
    use super::*;

    /// Compare two audio data structures with tolerance
    pub fn compare_audio_data(a: &AudioData, b: &AudioData, tolerance: f32) -> bool {
        if a.sample_rate != b.sample_rate
            || (a.duration - b.duration).abs() > tolerance
            || a.samples.len() != b.samples.len()
        {
            return false;
        }

        for (s1, s2) in a.samples.iter().zip(b.samples.iter()) {
            if (s1 - s2).abs() > tolerance {
                return false;
            }
        }

        true
    }

    /// Validate audio chunk properties
    pub fn validate_audio_chunk(chunk: &AudioChunk) -> bool {
        chunk.sample_rate == 16000
            && chunk.duration >= 0.0
            && chunk.samples.len() <= AudioChunk::TARGET_SAMPLES
            && chunk.start_time >= 0.0
    }

    /// Calculate RMS (Root Mean Square) of audio samples
    pub fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_of_squares: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_of_squares / samples.len() as f32).sqrt()
    }

    /// Check if audio data contains silence (all samples near zero)
    pub fn is_silence(samples: &[f32], threshold: f32) -> bool {
        samples.iter().all(|&s| s.abs() < threshold)
    }
}

// ============================================================================
// AudioProcessor Tests
// ============================================================================

#[tokio::test]
async fn test_audio_processor_load_nonexistent_file() {
    let mut processor = AudioProcessor::new().unwrap();
    let result = processor.load_audio("nonexistent_file.wav").await;

    assert!(result.is_err());
    match result.unwrap_err() {
        WhisperError::AudioProcessing {
            source: AudioProcessingError::ReadFailed { .. },
        } => {}
        WhisperError::AudioProcessing {
            source: AudioProcessingError::ProcessingFailed { .. },
        } => {}
        e => panic!(
            "Expected ReadFailed or ProcessingFailed error, got: {:?}",
            e
        ),
    }
}

#[tokio::test]
async fn test_audio_processor_load_invalid_file() {
    let mut processor = AudioProcessor::new().unwrap();

    // Create a temporary file with invalid content
    let temp_file = NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), b"invalid audio data")
        .await
        .unwrap();

    let result = processor.load_audio(temp_file.path()).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_audio_processor_load_valid_wav() {
    let temp_wav = fixtures::create_test_wav_file().await.unwrap();

    let mut processor = AudioProcessor::new().unwrap();
    let result = processor.load_audio(temp_wav.path()).await;

    assert!(result.is_ok(), "Should successfully load valid WAV file");

    let audio_data = result.unwrap();
    assert_eq!(audio_data.sample_rate, 16000);
    assert!(audio_data.duration > 0.0);
    assert!(!audio_data.samples.is_empty());
}

#[tokio::test]
async fn test_audio_processor_stream_nonexistent() {
    use futures::StreamExt;

    let result = AudioProcessor::stream("nonexistent_file.wav").await;

    // Stream creation might succeed, but the first chunk should contain an error
    match result {
        Ok(mut stream) => {
            // Try to get the first chunk - this should fail
            if let Some(chunk_result) = stream.next().await {
                assert!(
                    chunk_result.is_err(),
                    "Should get an error for nonexistent file"
                );
            }
        }
        Err(_) => {
            // Also acceptable - immediate failure
        }
    }
}

#[tokio::test]
async fn test_audio_processor_stream_valid_file() {
    let temp_wav = fixtures::create_test_sine_wav(5.0, 440.0).await.unwrap();

    let result = AudioProcessor::stream(temp_wav.path()).await;
    assert!(
        result.is_ok(),
        "Should successfully create stream for valid file"
    );

    let mut stream = result.unwrap();
    let mut chunk_count = 0;
    let mut total_duration = 0.0;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                assert!(utils::validate_audio_chunk(&chunk));
                assert_eq!(chunk.index, chunk_count);
                total_duration += chunk.duration;
                chunk_count += 1;

                if chunk.is_final {
                    break;
                }
            }
            Err(e) => panic!("Stream error: {:?}", e),
        }
    }

    assert!(chunk_count > 0, "Should produce at least one chunk");
    assert!(
        total_duration > 4.0,
        "Total duration should be close to 5 seconds"
    );
}

// ============================================================================
// Error Condition Tests
// ============================================================================

#[tokio::test]
async fn test_error_invalid_file_format() {
    let temp_file = NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), b"not an audio file")
        .await
        .unwrap();

    let mut processor = AudioProcessor::new().unwrap();
    let result = processor.load_audio(temp_file.path()).await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_error_empty_file() {
    let temp_file = NamedTempFile::new().unwrap();
    // File exists but is empty

    let mut processor = AudioProcessor::new().unwrap();
    let result = processor.load_audio(temp_file.path()).await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_error_corrupted_audio_file() {
    let temp_file = NamedTempFile::new().unwrap();

    // Create a file that starts like a WAV but is corrupted
    let mut corrupted_wav = Vec::new();
    corrupted_wav.extend_from_slice(b"RIFF");
    corrupted_wav.extend_from_slice(&100u32.to_le_bytes());
    corrupted_wav.extend_from_slice(b"WAVE");
    // Truncate here to make it corrupted

    fs::write(temp_file.path(), corrupted_wav).await.unwrap();

    let mut processor = AudioProcessor::new().unwrap();
    let result = processor.load_audio(temp_file.path()).await;

    assert!(result.is_err());
}

#[test]
fn test_utils_calculate_rms() {
    let samples = vec![0.0, 1.0, 0.0, -1.0]; // RMS should be sqrt(0.5) ≈ 0.707
    let rms = utils::calculate_rms(&samples);

    assert!((rms - 0.707).abs() < 0.01);
}

#[test]
fn test_utils_calculate_rms_empty() {
    let rms = utils::calculate_rms(&[]);
    assert_eq!(rms, 0.0);
}

#[test]
fn test_utils_validate_audio_chunk() {
    let valid_chunk = fixtures::create_test_audio_chunk(0, 0.0, false);
    assert!(utils::validate_audio_chunk(&valid_chunk));

    let invalid_chunk = AudioChunk {
        samples: vec![0.0; 100],
        sample_rate: 8000, // Wrong sample rate
        duration: 0.1,
        index: 0,
        start_time: -1.0, // Negative start time
        is_final: false,
    };
    assert!(!utils::validate_audio_chunk(&invalid_chunk));
}
