//! Integration tests for purr-core

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

    assert_eq!(config.language, Some("en".into()));
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
        WhisperError::AudioProcessing { .. } | WhisperError::Configuration { .. } => {}
        e => panic!(
            "Expected AudioProcessing or Configuration error, got: {}",
            e
        ),
    }
}

/// Test transcription with different configurations on a known sample
#[tokio::test]
async fn test_transcription_configurations() {
    let sample_path = "../../samples/jfk.wav";

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
    let sample_path = "../../samples/jfk.wav";

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
        WhisperError::AudioProcessing { .. } | WhisperError::Io { .. } => {
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
