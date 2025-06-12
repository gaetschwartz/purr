//! Integration tests for whisper-ui-core

use whisper_ui_core::*;
use tempfile::{NamedTempFile, TempDir};
use std::fs;
use std::io::Write;

/// Test transcription configuration
#[tokio::test]
async fn test_transcription_config() {
    let config = TranscriptionConfig::new()
        .with_language("en")
        .with_gpu(false)
        .with_threads(2)
        .with_sample_rate(16000);

    assert_eq!(config.language, Some("en".to_string()));
    assert!(!config.use_gpu);
    assert_eq!(config.num_threads, Some(2));
    assert_eq!(config.sample_rate, 16000);
}

/// Test error handling for missing files
#[tokio::test]
async fn test_missing_audio_file() {
    let config = TranscriptionConfig::new().with_gpu(false);
    let result = transcribe_audio_file("nonexistent_file.wav", Some(config)).await;
    
    assert!(result.is_err());
    match result.unwrap_err() {
        WhisperError::AudioProcessing(_) => (),
        _ => panic!("Expected AudioProcessing error"),
    }
}

/// Test audio processor creation
#[tokio::test]
async fn test_audio_processor() {
    let processor = AudioProcessor::new();
    // Just test that we can create it
    assert!(true);
}

/// Test serialization of transcription results
#[test]
fn test_transcription_result_serialization() {
    let segment = TranscriptionSegment {
        text: "Hello world".to_string(),
        start: 0.0,
        end: 2.5,
        confidence: Some(0.95),
        words: None,
    };

    let result = TranscriptionResult {
        text: "Hello world".to_string(),
        language: Some("en".to_string()),
        segments: vec![segment],
        processing_time: 1.5,
        audio_duration: 2.5,
    };

    // Test JSON serialization
    let json = serde_json::to_string(&result).unwrap();
    let deserialized: TranscriptionResult = serde_json::from_str(&json).unwrap();

    assert_eq!(result.text, deserialized.text);
    assert_eq!(result.language, deserialized.language);
    assert_eq!(result.segments.len(), deserialized.segments.len());
    assert_eq!(result.processing_time, deserialized.processing_time);
    assert_eq!(result.audio_duration, deserialized.audio_duration);
}

/// Test configuration defaults
#[test]
fn test_config_defaults() {
    let config = TranscriptionConfig::default();
    
    assert!(config.model_path.is_none());
    assert!(config.language.is_none());
    assert!(config.use_gpu);
    assert_eq!(config.sample_rate, 16000);
    assert_eq!(config.temperature, 0.0);
    assert!(config.output_format.include_timestamps);
    assert!(!config.output_format.word_timestamps);
    assert!(!config.output_format.include_confidence);
}

/// Test error types
#[test]
fn test_error_types() {
    let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
    let whisper_error = WhisperError::from(io_error);
    
    match whisper_error {
        WhisperError::Io(_) => (),
        _ => panic!("Expected IO error"),
    }
    
    let audio_error = WhisperError::AudioProcessing("Test error".to_string());
    assert!(audio_error.to_string().contains("Audio processing error"));
    
    let model_error = WhisperError::WhisperModel("Test model error".to_string());
    assert!(model_error.to_string().contains("Whisper model error"));
}
