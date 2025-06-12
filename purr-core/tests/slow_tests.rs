//! Integration tests for whisper-ui-core

use purr_core::*;
use rstest::rstest;
use std::path::PathBuf;

/// Test transcription with all sample files
#[rstest]
#[tokio::test]
async fn test_transcribe_sample_files(#[files("../samples/*")] sample_path: PathBuf) {
    // Skip README.md and other non-audio files
    let extension = sample_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");

    match extension.to_lowercase().as_str() {
        "wav" | "mp3" | "ogg" | "flac" => {
            println!("Testing transcription of: {}", sample_path.display());

            let config = TranscriptionConfig::new()
                .with_gpu(false) // Use CPU for consistent test results
                .with_sample_rate(16000);

            let result = transcribe_audio_file(&sample_path, Some(config)).await;

            match result {
                Ok(transcription) => {
                    // Basic validation of transcription result
                    assert!(
                        !transcription.text.is_empty(),
                        "Transcription should not be empty for {}",
                        sample_path.display()
                    );
                    assert!(
                        transcription.audio_duration > 0.0,
                        "Audio duration should be positive for {}",
                        sample_path.display()
                    );
                    assert!(
                        transcription.processing_time > 0.0,
                        "Processing time should be positive for {}",
                        sample_path.display()
                    );
                    assert!(
                        !transcription.segments.is_empty(),
                        "Should have at least one segment for {}",
                        sample_path.display()
                    );

                    // Validate segments
                    for (i, segment) in transcription.segments.iter().enumerate() {
                        assert!(
                            !segment.text.is_empty(),
                            "Segment {} text should not be empty for {}",
                            i,
                            sample_path.display()
                        );
                        assert!(
                            segment.start >= 0.0,
                            "Segment {} start time should be non-negative for {}",
                            i,
                            sample_path.display()
                        );
                        assert!(
                            segment.end >= segment.start,
                            "Segment {} end time should be >= start time for {}",
                            i,
                            sample_path.display()
                        );
                    }

                    // Validate that segments are in chronological order
                    for i in 1..transcription.segments.len() {
                        assert!(
                            transcription.segments[i].start >= transcription.segments[i - 1].start,
                            "Segments should be in chronological order for {}",
                            sample_path.display()
                        );
                    }

                    println!("✓ Successfully transcribed {} ({:.2}s audio, {:.2}s processing): \"{}...\"", 
                            sample_path.file_name().unwrap().to_string_lossy(),
                            transcription.audio_duration,
                            transcription.processing_time,
                            transcription.text.chars().take(50).collect::<String>());
                }
                Err(e) => {
                    // For tests, we should have at least one model available
                    // If no model is found, that's a test setup issue, not a code issue
                    if e.to_string().contains("No Whisper model found") {
                        println!(
                            "⚠ Skipping {} - no model available for testing",
                            sample_path.display()
                        );
                        return; // Skip this test rather than fail
                    } else {
                        panic!("Failed to transcribe {}: {}", sample_path.display(), e);
                    }
                }
            }
        }
        "md" => {
            println!("⏭ Skipping README file: {}", sample_path.display());
            return; // Skip non-audio files
        }
        _ => {
            println!(
                "⚠ Unknown file type for: {} (extension: {})",
                sample_path.display(),
                extension
            );
            return; // Skip unknown file types
        }
    }
}
