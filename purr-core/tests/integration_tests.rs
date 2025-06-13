//! Integration tests for purr-core

use purr_core::transcription::TranscriptionSegment;
use purr_core::*;
use rstest::rstest;
use std::path::Path;

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
    let result = transcribe_file_sync("nonexistent_file.wav", Some(config)).await;

    assert!(result.is_err());
    match result.unwrap_err() {
        WhisperError::AudioProcessing(_) | WhisperError::Configuration(_) => {}
        e => panic!(
            "Expected AudioProcessing or Configuration error, got: {}",
            e
        ),
    }
}

/// Test audio processor creation
#[tokio::test]
async fn test_audio_processor() {
    // Just test that we can create it
    let _processor = AudioProcessor::new();
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

/// Test transcription with different configurations on a known sample
#[tokio::test]
async fn test_transcription_configurations() {
    let sample_path = "../samples/jfk.wav";

    // Test with different configurations
    let configs = vec![
        TranscriptionConfig::new()
            .with_gpu(false)
            .with_language("en"),
        TranscriptionConfig::new().with_gpu(false).with_threads(1),
        {
            let mut config = TranscriptionConfig::new()
                .with_gpu(false)
                .with_sample_rate(16000);
            config.temperature = 0.1;
            config
        },
    ];

    for (i, config) in configs.into_iter().enumerate() {
        println!("Testing configuration {}", i + 1);

        let result = transcribe_file_sync(sample_path, Some(config)).await;

        match result {
            Ok(transcription) => {
                assert!(!transcription.text.is_empty());
                assert!(transcription.audio_duration > 0.0);
                println!(
                    "✓ Configuration {} successful: \"{}...\"",
                    i + 1,
                    transcription.text.chars().take(30).collect::<String>()
                );
            }
            Err(e) => {
                if e.to_string().contains("No Whisper model found") {
                    println!("⚠ Skipping configuration {} - no model available", i + 1);
                    continue;
                } else {
                    panic!("Configuration {} failed: {}", i + 1, e);
                }
            }
        }
    }
}

/// Test transcription output formats
#[tokio::test]
async fn test_transcription_output_formats() {
    let sample_path = "../samples/jfk.wav";

    // Skip if sample doesn't exist
    if !Path::new(sample_path).exists() {
        println!("⏭ Skipping output format test - sample file not found");
        return;
    }

    let mut config = TranscriptionConfig::new()
        .with_gpu(false)
        .with_language("en");

    // Test with timestamps enabled
    config.output_format.include_timestamps = true;
    config.output_format.word_timestamps = true;

    let result = transcribe_file_sync(sample_path, Some(config)).await;

    match result {
        Ok(transcription) => {
            assert!(!transcription.text.is_empty());
            assert!(!transcription.segments.is_empty());

            // Validate timestamp format
            for segment in &transcription.segments {
                assert!(segment.start >= 0.0);
                assert!(segment.end >= segment.start);
            }

            println!(
                "✓ Timestamps test successful with {} segments",
                transcription.segments.len()
            );
        }
        Err(e) => {
            if e.to_string().contains("No Whisper model found") {
                println!("⚠ Skipping timestamps test - no model available");
                return;
            } else {
                panic!("Timestamps test failed: {}", e);
            }
        }
    }
}

/// Test error handling with invalid files
#[rstest]
#[case("nonexistent.wav")]
#[case("../../Cargo.toml")] // Valid file but not audio
#[tokio::test]
async fn test_transcription_error_handling(#[case] invalid_path: &str) {
    let config = TranscriptionConfig::new().with_gpu(false);
    let result = transcribe_file_sync(invalid_path, Some(config)).await;

    assert!(
        result.is_err(),
        "Should fail for invalid file: {}",
        invalid_path
    );

    let error = result.unwrap_err();
    match error {
        WhisperError::AudioProcessing(_) | WhisperError::Io(_) => {
            println!(
                "✓ Correctly handled invalid file: {} -> {}",
                invalid_path, error
            );
        }
        _ => {
            println!("⚠ Unexpected error type for {}: {}", invalid_path, error);
        }
    }
}
