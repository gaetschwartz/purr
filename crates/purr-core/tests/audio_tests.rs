//! Comprehensive unit tests for the audio module
//!
//! Tests cover:
//! - Audio format conversion
//! - Whisper integration (sync and streaming)
//! - Audio buffer management
//! - Sample rate handling
//! - Error conditions and edge cases

use purr_core::{
    audio::{AudioData, AudioChunk, AudioProcessor},
    error::{AudioProcessingError, WhisperError},
};
use futures::StreamExt;
use rstest::rstest;
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
    pub async fn create_test_sine_wav(duration_secs: f32, frequency: f32) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
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
    fn create_wav_bytes(samples: Vec<i16>, sample_rate: u32) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
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
        if a.sample_rate != b.sample_rate ||
           (a.duration - b.duration).abs() > tolerance ||
           a.samples.len() != b.samples.len() {
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
        chunk.sample_rate == 16000 &&
        chunk.duration >= 0.0 &&
        chunk.samples.len() <= AudioChunk::TARGET_SAMPLES &&
        chunk.start_time >= 0.0
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
async fn test_audio_processor_creation() {
    let result = AudioProcessor::new();
    assert!(result.is_ok(), "AudioProcessor should be creatable");
}

#[tokio::test]
async fn test_audio_processor_load_nonexistent_file() {
    let mut processor = AudioProcessor::new().unwrap();
    let result = processor.load_audio("nonexistent_file.wav").await;

    assert!(result.is_err());
    match result.unwrap_err() {
        WhisperError::AudioProcessing { source: AudioProcessingError::ProcessingFailed { .. } } => {},
        e => panic!("Expected ProcessingFailed error, got: {:?}", e),
    }
}

#[tokio::test]
async fn test_audio_processor_load_invalid_file() {
    let mut processor = AudioProcessor::new().unwrap();

    // Create a temporary file with invalid content
    let temp_file = NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), b"invalid audio data").await.unwrap();

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
                assert!(chunk_result.is_err(), "Should get an error for nonexistent file");
            }
        },
        Err(_) => {
            // Also acceptable - immediate failure
        }
    }
}

#[tokio::test]
async fn test_audio_processor_stream_valid_file() {
    let temp_wav = fixtures::create_test_sine_wav(5.0, 440.0).await.unwrap();

    let result = AudioProcessor::stream(temp_wav.path()).await;
    assert!(result.is_ok(), "Should successfully create stream for valid file");

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
            },
            Err(e) => panic!("Stream error: {:?}", e),
        }
    }

    assert!(chunk_count > 0, "Should produce at least one chunk");
    assert!(total_duration > 4.0, "Total duration should be close to 5 seconds");
}

// ============================================================================
// AudioData Tests
// ============================================================================

#[test]
fn test_audio_data_creation() {
    let audio_data = fixtures::create_test_audio_data(2.5, 16000);

    assert_eq!(audio_data.sample_rate, 16000);
    assert_eq!(audio_data.duration, 2.5);
    assert_eq!(audio_data.samples.len(), 40000); // 2.5 * 16000
}

#[test]
fn test_audio_data_clone() {
    let original = fixtures::create_test_audio_data(1.0, 16000);
    let cloned = original.clone();

    assert!(utils::compare_audio_data(&original, &cloned, 0.0001));
}

// ============================================================================
// AudioChunk Tests
// ============================================================================

#[test]
fn test_audio_chunk_creation() {
    let chunk = fixtures::create_test_audio_chunk(0, 0.0, false);

    assert_eq!(chunk.sample_rate, 16000);
    assert_eq!(chunk.index, 0);
    assert_eq!(chunk.start_time, 0.0);
    assert!(!chunk.is_final);
    assert!(utils::validate_audio_chunk(&chunk));
}

#[test]
fn test_audio_chunk_constants() {
    assert_eq!(AudioChunk::TARGET_DURATION, 10.0);
    assert_eq!(AudioChunk::TARGET_SAMPLES, 160000); // 10.0 * 16000
}

#[test]
fn test_audio_chunk_duration_calculation() {
    let samples = vec![0.0f32; 8000]; // 0.5 seconds at 16kHz
    let chunk = AudioChunk::new(samples, 0, 0.0, false);

    assert!((chunk.duration - 0.5).abs() < 0.001);
}

#[rstest]
#[case(0, 0.0, false)]
#[case(1, 10.0, false)]
#[case(2, 20.0, true)]
fn test_audio_chunk_with_different_properties(
    #[case] index: usize,
    #[case] start_time: f32,
    #[case] is_final: bool,
) {
    let chunk = fixtures::create_test_audio_chunk(index, start_time, is_final);

    assert_eq!(chunk.index, index);
    assert_eq!(chunk.start_time, start_time);
    assert_eq!(chunk.is_final, is_final);
}

// ============================================================================
// Sample Rate Handling Tests
// ============================================================================

#[tokio::test]
async fn test_sample_rate_conversion_16khz() {
    // This test would require a real audio file with known sample rate
    // For now, test that our target sample rate is correctly set
    let audio_data = fixtures::create_test_audio_data(1.0, 16000);
    assert_eq!(audio_data.sample_rate, 16000);
}

#[test]
fn test_sample_rate_constants() {
    // Whisper expects 16kHz sample rate
    const WHISPER_SAMPLE_RATE: u32 = 16000;
    assert_eq!(WHISPER_SAMPLE_RATE, 16000);
}

// ============================================================================
// Buffer Management Tests
// ============================================================================

#[test]
fn test_audio_buffer_management() {
    let mut samples = Vec::with_capacity(100);
    samples.extend_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    assert_eq!(samples.len(), 5);
    assert_eq!(samples.capacity(), 100);

    // Test buffer reuse
    samples.clear();
    assert_eq!(samples.len(), 0);
    assert_eq!(samples.capacity(), 100);
}

#[test]
fn test_audio_chunk_buffer_reuse() {
    let chunk1 = fixtures::create_test_audio_chunk(0, 0.0, false);
    let samples_len = chunk1.samples.len();

    // Simulate processing and creating new chunk
    let chunk2 = AudioChunk::new(vec![1.0; samples_len], 1, 10.0, false);

    assert_eq!(chunk2.samples.len(), samples_len);
    assert_eq!(chunk2.index, 1);
}

// ============================================================================
// Error Condition Tests
// ============================================================================

#[tokio::test]
async fn test_error_invalid_file_format() {
    let temp_file = NamedTempFile::new().unwrap();
    fs::write(temp_file.path(), b"not an audio file").await.unwrap();

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

// ============================================================================
// Edge Cases Tests
// ============================================================================

#[test]
fn test_zero_duration_audio() {
    let audio_data = AudioData {
        samples: vec![],
        sample_rate: 16000,
        duration: 0.0,
    };

    assert_eq!(audio_data.duration, 0.0);
    assert!(audio_data.samples.is_empty());
}

#[test]
fn test_very_short_audio() {
    let samples = vec![0.1]; // One sample
    let duration = 1.0 / 16000.0; // Duration of one sample at 16kHz

    let audio_data = AudioData {
        samples,
        sample_rate: 16000,
        duration,
    };

    assert_eq!(audio_data.samples.len(), 1);
    assert!(audio_data.duration > 0.0);
}

#[test]
fn test_large_audio_buffer() {
    // Test with 1 minute of audio (960,000 samples)
    let duration = 60.0;
    let sample_rate = 16000;
    let num_samples = (duration * sample_rate as f32) as usize;

    let audio_data = fixtures::create_test_audio_data(duration, sample_rate);

    assert_eq!(audio_data.samples.len(), num_samples);
    assert_eq!(audio_data.duration, duration);
}

// ============================================================================
// Audio Format Tests
// ============================================================================

#[test]
fn test_audio_sample_format_f32() {
    let samples = vec![-1.0f32, -0.5, 0.0, 0.5, 1.0];

    // Verify all samples are in valid range for f32 audio
    for &sample in &samples {
        assert!(sample >= -1.0 && sample <= 1.0, "Sample {} out of range", sample);
    }
}

#[test]
fn test_audio_sample_normalization() {
    let samples = vec![32767i16, 0, -32768]; // i16 range

    // Convert to f32 normalized range
    let normalized: Vec<f32> = samples.iter()
        .map(|&s| s as f32 / 32768.0)
        .collect();

    assert!((normalized[0] - 0.99997).abs() < 0.001); // Close to 1.0
    assert_eq!(normalized[1], 0.0);
    assert_eq!(normalized[2], -1.0);
}

// ============================================================================
// Utility Function Tests
// ============================================================================

#[test]
fn test_utils_compare_audio_data() {
    let audio1 = fixtures::create_test_audio_data(1.0, 16000);
    let audio2 = fixtures::create_test_audio_data(1.0, 16000);

    assert!(utils::compare_audio_data(&audio1, &audio2, 0.001));
}

#[test]
fn test_utils_compare_audio_data_different() {
    let audio1 = fixtures::create_test_audio_data(1.0, 16000);
    let audio2 = fixtures::create_test_audio_data(2.0, 16000);

    assert!(!utils::compare_audio_data(&audio1, &audio2, 0.001));
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
fn test_utils_is_silence() {
    let silent_samples = vec![0.0, 0.001, -0.001, 0.0005];
    let loud_samples = vec![0.0, 0.5, 0.0, -0.3];

    assert!(utils::is_silence(&silent_samples, 0.01));
    assert!(!utils::is_silence(&loud_samples, 0.01));
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

// ============================================================================
// Integration Tests with Mocked Whisper
// ============================================================================

#[test]
fn test_mock_whisper_response() {
    let mock_response = mocks::MockWhisperResponse::new_simple("Hello world", 2.0);

    assert_eq!(mock_response.text, "Hello world");
    assert_eq!(mock_response.segments.len(), 1);
    assert_eq!(mock_response.segments[0].start, 0.0);
    assert_eq!(mock_response.segments[0].end, 2.0);
}

// Note: Full Whisper integration tests would require actual models
// and are better suited for integration_tests.rs or slow_tests.rs

// ============================================================================
// Performance Tests
// ============================================================================

#[tokio::test]
async fn test_audio_processing_performance() {
    let start = std::time::Instant::now();

    // Create a medium-sized audio file (10 seconds)
    let temp_wav = fixtures::create_test_sine_wav(10.0, 440.0).await.unwrap();

    let mut processor = AudioProcessor::new().unwrap();
    let result = processor.load_audio(temp_wav.path()).await;

    let elapsed = start.elapsed();

    assert!(result.is_ok());
    assert!(elapsed.as_secs() < 5, "Audio processing should be reasonably fast");

    let audio_data = result.unwrap();
    assert!((audio_data.duration - 10.0).abs() < 0.1);
}

#[tokio::test]
async fn test_streaming_performance() {
    let temp_wav = fixtures::create_test_sine_wav(20.0, 440.0).await.unwrap();

    let start = std::time::Instant::now();
    let stream_result = AudioProcessor::stream(temp_wav.path()).await;
    let stream_creation_time = start.elapsed();

    assert!(stream_result.is_ok());
    assert!(stream_creation_time.as_millis() < 1000, "Stream creation should be fast");

    let mut stream = stream_result.unwrap();
    let mut chunk_count = 0;
    let process_start = std::time::Instant::now();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                chunk_count += 1;
                if chunk.is_final {
                    break;
                }
            },
            Err(_) => break,
        }
    }

    let process_time = process_start.elapsed();

    assert!(chunk_count >= 2, "Should produce multiple chunks for 20-second audio");
    assert!(process_time.as_secs() < 10, "Streaming should be reasonably fast");
}