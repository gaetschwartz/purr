//! Whisper integration tests for audio processing
//!
//! These tests require a Whisper model to be available and test the full
//! audio-to-transcription pipeline including both sync and streaming modes.

use futures::{Stream, StreamExt};
use purr_common::progress::ProgressPhase;
use purr_core::{
    ModelManager, StreamEvent, StreamingChunk, StreamingTranscription,
    audio::{AudioData, AudioProcessor},
    config::TranscriptionConfig,
    transcribe_file_stream, transcribe_file_sync,
    whisper::{
        SyncTranscriptionResult, WhisperTranscriber, streaming::StreamWhisperTranscriber,
        sync::SyncWhisperTranscriber,
    },
};
use rstest::rstest;
use std::path::PathBuf;
use tempfile::NamedTempFile;
use tokio::fs;

/// The chunks of a stream, without its progress events.
fn chunks_of(
    stream: StreamingTranscription,
) -> impl Stream<Item = purr_core::Result<StreamingChunk>> + Unpin {
    stream
        .filter_map(|event| std::future::ready(event.map(StreamEvent::into_chunk).transpose()))
        .boxed()
}

/// Test fixtures for Whisper integration testing
mod whisper_fixtures {
    use super::*;

    /// Create a test WAV file with sine wave for transcription testing
    pub async fn create_test_audio_with_tone(
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
            samples.push((sample * 16383.0) as i16); // Scale to i16 range but not full scale
        }

        let wav_data = create_wav_bytes(samples, sample_rate)?;
        fs::write(temp_file.path(), wav_data).await?;

        Ok(temp_file)
    }

    /// Create a short audio file for quick testing
    pub async fn create_short_test_audio() -> Result<NamedTempFile, Box<dyn std::error::Error>> {
        create_test_audio_with_tone(3.0, 440.0).await // 3 seconds, 440Hz tone
    }

    /// Create a longer audio file for streaming tests
    pub async fn create_streaming_test_audio() -> Result<NamedTempFile, Box<dyn std::error::Error>>
    {
        create_test_audio_with_tone(25.0, 880.0).await // 25 seconds, 880Hz tone
    }

    /// Helper to create minimal WAV file bytes
    pub fn create_wav_bytes(
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

    /// Check if a Whisper model is available for testing
    pub async fn is_model_available() -> bool {
        let model_manager = match ModelManager::new() {
            Ok(manager) => manager,
            Err(_) => return false,
        };
        model_manager.find_default_model().await.is_ok()
    }
}

/// Validation utilities for transcription results
mod validation {
    use super::*;

    /// Validate sync transcription result structure
    pub fn validate_sync_result(result: &SyncTranscriptionResult) -> bool {
        // Basic structure validation
        result.processing_time > 0.0 &&
        result.audio_duration > 0.0 &&
        result.stats.processing_time > 0.0 &&
        result.stats.audio_duration > 0.0 &&
        // If segments exist, they should be in chronological order
        result.segments.windows(2).all(|w| w[0].start <= w[1].start) &&
        // Each segment should have valid timestamps
        result.segments.iter().all(|s| s.start >= 0.0 && s.end >= s.start) &&
        // Consistency checks: stats should match result data
        (result.stats.processing_time - result.processing_time).abs() < 0.001 &&
        (result.stats.audio_duration - result.audio_duration).abs() < 0.001
    }

    /// Validate that streaming chunks are properly ordered and structured
    pub fn validate_streaming_chunks(chunks: &[purr_core::whisper::StreamingChunk]) -> bool {
        if chunks.is_empty() {
            return false;
        }

        // Check that chunks are in order
        let chunks_in_order = chunks
            .windows(2)
            .all(|w| w[0].chunk_index <= w[1].chunk_index);

        // Check that the last chunk is marked as final
        let last_is_final = chunks.last().is_some_and(|c| c.is_final);

        // Check that all chunks have valid timestamps
        let valid_timestamps = chunks.iter().all(|c| c.start >= 0.0 && c.end >= c.start);

        chunks_in_order && last_is_final && valid_timestamps
    }
}

// ============================================================================
// Sync Whisper Integration Tests
// ============================================================================

#[tokio::test]
async fn test_sync_whisper_transcriber_creation() {
    if !whisper_fixtures::is_model_available().await {
        println!("⏭ Skipping test - no Whisper model available");
        return;
    }

    let config = TranscriptionConfig::new()
        .with_gpu(false)
        .with_language("en");

    let result = SyncWhisperTranscriber::from_config(config).await;
    assert!(
        result.is_ok(),
        "Should be able to create SyncWhisperTranscriber"
    );
}

#[tokio::test]
async fn test_sync_whisper_transcribe_audio_data() {
    if !whisper_fixtures::is_model_available().await {
        println!("⏭ Skipping test - no Whisper model available");
        return;
    }

    let config = TranscriptionConfig::new()
        .with_gpu(false)
        .with_language("en");

    let transcriber = SyncWhisperTranscriber::from_config(config).await.unwrap();

    // Create test audio data (3 seconds of silence)
    let audio_data = AudioData {
        samples: vec![0.0; 48000], // 3 seconds at 16kHz
        sample_rate: 16000,
        duration: 3.0,
    };

    let result = transcriber.transcribe(audio_data).await;
    assert!(result.is_ok(), "Transcription should succeed");

    let transcription = result.unwrap();
    assert!(validation::validate_sync_result(&transcription));
    // For silence, we might get empty or minimal transcription
    assert!(transcription.audio_duration > 2.5);
}

#[tokio::test]
async fn test_sync_whisper_with_generated_audio() {
    if !whisper_fixtures::is_model_available().await {
        println!("⏭ Skipping test - no Whisper model available");
        return;
    }

    let temp_audio = whisper_fixtures::create_short_test_audio().await.unwrap();

    let config = TranscriptionConfig::new()
        .with_gpu(false)
        .with_language("en")
        .with_verbose(false);

    let audio_data = AudioProcessor::load_audio(temp_audio.path()).await.unwrap();

    let transcriber = SyncWhisperTranscriber::from_config(config).await.unwrap();
    let result = transcriber.transcribe(audio_data).await;

    assert!(
        result.is_ok(),
        "Should successfully transcribe generated audio"
    );

    let transcription = result.unwrap();
    assert!(validation::validate_sync_result(&transcription));
    assert!((transcription.audio_duration - 3.0).abs() < 0.5); // Should be ~3 seconds
}

#[tokio::test]
async fn test_sync_transcription_high_level_api() {
    if !whisper_fixtures::is_model_available().await {
        println!("⏭ Skipping test - no Whisper model available");
        return;
    }

    let temp_audio = whisper_fixtures::create_short_test_audio().await.unwrap();

    let config = TranscriptionConfig::new()
        .with_gpu(false)
        .with_language("en");

    let result = transcribe_file_sync(temp_audio.path(), Some(config)).await;
    assert!(result.is_ok(), "High-level sync API should work");

    let transcription = result.unwrap();
    assert!(validation::validate_sync_result(&transcription));
}

// ============================================================================
// Streaming Whisper Integration Tests
// ============================================================================

#[tokio::test]
async fn test_streaming_whisper_transcriber_creation() {
    if !whisper_fixtures::is_model_available().await {
        println!("⏭ Skipping test - no Whisper model available");
        return;
    }

    let config = TranscriptionConfig::new()
        .with_gpu(false)
        .with_language("en");

    let result = StreamWhisperTranscriber::from_config(config).await;
    assert!(
        result.is_ok(),
        "Should be able to create StreamWhisperTranscriber"
    );
}

#[tokio::test]
async fn test_streaming_whisper_with_audio_stream() {
    if !whisper_fixtures::is_model_available().await {
        println!("⏭ Skipping test - no Whisper model available");
        return;
    }

    let temp_audio = whisper_fixtures::create_streaming_test_audio()
        .await
        .unwrap();

    let config = TranscriptionConfig::new()
        .with_gpu(false)
        .with_language("en");

    let transcriber = StreamWhisperTranscriber::from_config(config).await.unwrap();
    let audio_stream = AudioProcessor::stream_file(temp_audio.path())
        .await
        .unwrap();

    let result = transcriber.transcribe(audio_stream).await;
    assert!(result.is_ok(), "Streaming transcription should succeed");

    let mut streaming_result = chunks_of(result.unwrap());
    let mut chunks = Vec::new();
    let mut final_stats = None;

    while let Some(chunk_result) = streaming_result.next().await {
        match chunk_result {
            Ok(chunk) => {
                // Validate chunk structure
                assert!(chunk.start >= 0.0);
                assert!(chunk.end >= chunk.start);
                assert!(!chunk.text.is_empty() || chunk.chunk_index == 0); // First chunk might be empty

                if let Some(ref stats) = chunk.final_stats {
                    final_stats = Some(stats.clone());
                }

                chunks.push(chunk.clone());

                if chunk.is_final {
                    break;
                }
            }
            Err(e) => panic!("Streaming error: {e:?}"),
        }
    }

    assert!(!chunks.is_empty(), "Should produce at least one chunk");
    assert!(validation::validate_streaming_chunks(&chunks));
    assert!(final_stats.is_some(), "Should have final statistics");

    let stats = final_stats.unwrap();
    assert!(stats.audio_duration > 20.0); // Should be ~25 seconds
    assert!(stats.processing_time > 0.0);
}

#[tokio::test]
async fn test_streaming_transcription_high_level_api() {
    if !whisper_fixtures::is_model_available().await {
        println!("⏭ Skipping test - no Whisper model available");
        return;
    }

    let temp_audio = whisper_fixtures::create_streaming_test_audio()
        .await
        .unwrap();

    let config = TranscriptionConfig::new()
        .with_gpu(false)
        .with_language("en");

    let result = transcribe_file_stream(temp_audio.path(), config).await;
    assert!(result.is_ok(), "High-level streaming API should work");

    let mut streaming_result = chunks_of(result.unwrap());
    let mut chunk_count = 0;
    let mut received_final = false;

    while let Some(chunk_result) = streaming_result.next().await {
        match chunk_result {
            Ok(chunk) => {
                chunk_count += 1;
                if chunk.is_final {
                    received_final = true;
                    // Verify we have final stats
                    assert!(chunk.final_stats.is_some(), "Final chunk should have stats");
                    break;
                }
            }
            Err(e) => panic!("Streaming error: {e:?}"),
        }
    }

    // With true streaming implementation, we expect multiple chunks as audio is processed progressively
    // The exact number depends on audio length and chunk size, but should be more than 1 for proper streaming
    assert!(chunk_count > 0, "Should produce at least one chunk");
    assert!(received_final, "Should receive final chunk");

    // For streaming audio (typically 25+ seconds), we should get multiple chunks
    if chunk_count == 1 {
        println!(
            "⚠ Warning: Only received 1 chunk - this might indicate streaming isn't working properly"
        );
    } else {
        println!("✓ Received {chunk_count} streaming chunks as expected");
    }
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[rstest]
#[case(true)] // With GPU
#[case(false)] // Without GPU
#[tokio::test]
async fn test_transcription_with_gpu_settings(#[case] use_gpu: bool) {
    if !whisper_fixtures::is_model_available().await {
        println!("⏭ Skipping test - no Whisper model available");
        return;
    }

    let temp_audio = whisper_fixtures::create_short_test_audio().await.unwrap();

    let config = TranscriptionConfig::new()
        .with_gpu(use_gpu)
        .with_language("en");

    let result = transcribe_file_sync(temp_audio.path(), Some(config)).await;

    // Test should succeed regardless of GPU availability
    match result {
        Ok(transcription) => {
            assert!(validation::validate_sync_result(&transcription));
            println!(
                "✓ GPU setting {} worked",
                if use_gpu { "enabled" } else { "disabled" }
            );
        }
        Err(e) => {
            // GPU might not be available, which is acceptable
            if use_gpu && e.to_string().contains("GPU") {
                println!("⚠ GPU not available, test skipped");
            } else {
                panic!("Unexpected error with GPU {use_gpu}: {e}");
            }
        }
    }
}

#[rstest]
#[case("en")]
#[case("auto")]
#[tokio::test]
async fn test_transcription_with_language_settings(#[case] language: &'static str) {
    if !whisper_fixtures::is_model_available().await {
        println!("⏭ Skipping test - no Whisper model available");
        return;
    }

    let temp_audio = whisper_fixtures::create_short_test_audio().await.unwrap();

    let mut config = TranscriptionConfig::new().with_gpu(false);

    if language != "auto" {
        config = config.with_language(language);
    }

    let result = transcribe_file_sync(temp_audio.path(), Some(config)).await;
    assert!(
        result.is_ok(),
        "Should work with language setting: {language}"
    );

    let transcription = result.unwrap();
    assert!(validation::validate_sync_result(&transcription));
}

#[rstest]
#[case(1)]
#[case(2)]
#[case(4)]
#[tokio::test]
async fn test_transcription_with_thread_settings(#[case] num_threads: usize) {
    if !whisper_fixtures::is_model_available().await {
        println!("⏭ Skipping test - no Whisper model available");
        return;
    }

    let temp_audio = whisper_fixtures::create_short_test_audio().await.unwrap();

    let config = TranscriptionConfig::new()
        .with_gpu(false)
        .with_threads(num_threads)
        .with_language("en");

    let result = transcribe_file_sync(temp_audio.path(), Some(config)).await;
    assert!(result.is_ok(), "Should work with {num_threads} threads");

    let transcription = result.unwrap();
    assert!(validation::validate_sync_result(&transcription));
}

// ============================================================================
// Performance Tests
// ============================================================================

#[tokio::test]
async fn test_transcription_performance_sync() {
    if !whisper_fixtures::is_model_available().await {
        println!("⏭ Skipping test - no Whisper model available");
        return;
    }

    let temp_audio = whisper_fixtures::create_short_test_audio().await.unwrap();

    let config = TranscriptionConfig::new()
        .with_gpu(false)
        .with_language("en");

    let start = std::time::Instant::now();
    let result = transcribe_file_sync(temp_audio.path(), Some(config)).await;
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Performance test should succeed");

    let transcription = result.unwrap();
    assert!(validation::validate_sync_result(&transcription));

    // Performance expectations (these are rough guidelines)
    assert!(
        elapsed.as_secs() < 30,
        "3-second audio should transcribe in under 30 seconds"
    );

    let real_time_factor = transcription.stats.real_time_factor();
    println!("Real-time factor: {real_time_factor:.2}x");

    // Real-time factor should be reasonable (but depends on hardware)
    if real_time_factor > 0.0 {
        assert!(
            real_time_factor < 100.0,
            "Real-time factor should be reasonable"
        );
    }
}

#[tokio::test]
async fn test_transcription_performance_streaming() {
    if !whisper_fixtures::is_model_available().await {
        println!("⏭ Skipping test - no Whisper model available");
        return;
    }

    let temp_audio = whisper_fixtures::create_streaming_test_audio()
        .await
        .unwrap();

    let config = TranscriptionConfig::new()
        .with_gpu(false)
        .with_language("en");

    let start = std::time::Instant::now();
    let result = transcribe_file_stream(temp_audio.path(), config).await;
    assert!(result.is_ok(), "Streaming performance test should succeed");

    let mut streaming_result = chunks_of(result.unwrap());
    let mut first_chunk_time = None;
    let mut _last_chunk_time = None;

    while let Some(chunk_result) = streaming_result.next().await {
        match chunk_result {
            Ok(chunk) => {
                if first_chunk_time.is_none() {
                    first_chunk_time = Some(std::time::Instant::now());
                }
                _last_chunk_time = Some(std::time::Instant::now());

                if chunk.is_final {
                    break;
                }
            }
            Err(e) => panic!("Streaming performance error: {e:?}"),
        }
    }

    let total_elapsed = start.elapsed();

    assert!(
        first_chunk_time.is_some(),
        "Should have received at least one chunk"
    );
    assert!(
        total_elapsed.as_secs() < 120,
        "25-second audio should stream in under 2 minutes"
    );

    println!(
        "Streaming performance: {:.2}s total",
        total_elapsed.as_secs_f64()
    );
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_whisper_error_handling_invalid_config() {
    if !whisper_fixtures::is_model_available().await {
        println!("⏭ Skipping test - no Whisper model available");
        return;
    }

    // Test with invalid model path
    let mut config = TranscriptionConfig::new()
        .with_gpu(false)
        .with_language("en");

    config.model_path = Some(PathBuf::from("definitely_does_not_exist.bin"));

    let result = SyncWhisperTranscriber::from_config(config).await;
    assert!(result.is_err(), "Should fail with invalid model path");
}

#[tokio::test]
async fn test_whisper_error_handling_empty_audio() {
    if !whisper_fixtures::is_model_available().await {
        println!("⏭ Skipping test - no Whisper model available");
        return;
    }

    let config = TranscriptionConfig::new()
        .with_gpu(false)
        .with_language("en");

    let transcriber = SyncWhisperTranscriber::from_config(config).await.unwrap();

    // Empty audio data
    let audio_data = AudioData {
        samples: vec![],
        sample_rate: 16000,
        duration: 0.0,
    };

    let result = transcriber.transcribe(audio_data).await;
    // This might succeed with empty transcription or fail - both are acceptable
    if let Ok(transcription) = result {
        assert_eq!(transcription.audio_duration, 0.0);
        assert!(transcription.text.is_empty() || transcription.text.trim().is_empty());
    } else {
        // Also acceptable - some implementations might reject empty audio
    }
}

// ============================================================================
// Regression Tests
// ============================================================================

/// How often the most repeated run of `n` words occurs in `text`, ignoring case and punctuation.
fn most_repeated_run(text: &str, n: usize) -> usize {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect();
    let mut counts = std::collections::HashMap::new();
    for run in words.windows(n) {
        *counts.entry(run).or_insert(0usize) += 1;
    }
    counts.into_values().max().unwrap_or(0)
}

#[tokio::test]
async fn test_streaming_does_not_loop_on_fast_speech() {
    let Ok(model_manager) = ModelManager::new() else {
        println!("⏭ Skipping test - no model directory");
        return;
    };
    let model_path = model_manager.models_dir().join("ggml-base.bin");
    if !model_path.exists() {
        println!("⏭ Skipping test - ggml-base.bin not available");
        return;
    }

    let mut config = TranscriptionConfig::new()
        .with_gpu(false)
        .with_language("en")
        .with_beam_size(1);
    config.model_path = Some(model_path);

    let sample = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/mm1.wav");
    let mut stream = chunks_of(transcribe_file_stream(&sample, config).await.unwrap());
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        text.push_str(&chunk.text);
        text.push(' ');
    }

    assert!(
        most_repeated_run(&text, 6) <= 2,
        "a run of six words repeats: {text}"
    );
}

/// How often the run of words `phrase` occurs in `text`, ignoring case and punctuation.
fn count_run(text: &str, phrase: &str) -> usize {
    let words = |text: &str| -> Vec<String> {
        text.split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|c| c.is_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect::<String>()
            })
            .filter(|word| !word.is_empty())
            .collect()
    };
    let (text, phrase) = (words(text), words(phrase));
    text.windows(phrase.len())
        .filter(|run| *run == phrase.as_slice())
        .count()
}

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

/// The sample spoken twice, two seconds apart, as a WAV file.
async fn spoken_twice(sample: &str) -> TestResult<NamedTempFile> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../samples")
        .join(sample);
    let audio = AudioProcessor::load_audio(&path).await?;
    let speech: Vec<i16> = audio
        .samples
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16)
        .collect();
    let silence = vec![0i16; audio.sample_rate as usize * 2];
    let samples = [speech.as_slice(), &silence, &speech].concat();
    let file = NamedTempFile::with_suffix(".wav")?;
    fs::write(
        file.path(),
        whisper_fixtures::create_wav_bytes(samples, audio.sample_rate)?,
    )
    .await?;
    Ok(file)
}

async fn stream_chunks(
    audio: &std::path::Path,
    config: TranscriptionConfig,
) -> TestResult<Vec<purr_core::StreamingChunk>> {
    let mut stream = chunks_of(transcribe_file_stream(audio, config).await?);
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk?);
    }
    Ok(chunks)
}

/// The voice activity model the streaming tests use: `PURR_VAD_MODEL`, or the default file in
/// the models directory
fn vad_model(model_manager: &ModelManager) -> Option<PathBuf> {
    std::env::var_os("PURR_VAD_MODEL")
        .map(PathBuf::from)
        .or_else(|| Some(model_manager.models_dir().join("ggml-silero-v6.2.0.bin")))
        .filter(|path| path.exists())
}

#[tokio::test]
async fn test_streaming_keeps_a_sentence_spoken_twice() {
    let Ok(model_manager) = ModelManager::new() else {
        println!("⏭ Skipping test - no model directory");
        return;
    };
    let model_path = model_manager.models_dir().join("ggml-base.bin");
    if !model_path.exists() {
        println!("⏭ Skipping test - ggml-base.bin not available");
        return;
    }
    let audio = spoken_twice("jfk.wav").await.unwrap();

    let mut config = TranscriptionConfig::new().with_gpu(false).with_beam_size(1);
    config.model_path = Some(model_path);
    let mut configs = vec![config.clone()];
    match vad_model(&model_manager) {
        Some(vad) => configs.push(config.with_vad_model(Some(vad))),
        None => println!("⏭ Skipping the voice activity variant - no Silero model"),
    }

    for config in configs {
        let chunks = stream_chunks(audio.path(), config).await.unwrap();
        assert!(validation::validate_streaming_chunks(&chunks));
        assert!(
            chunks
                .windows(2)
                .all(|pair| matches!(pair, [a, b] if a.end <= b.start + 0.01))
        );
        let stats = chunks.last().unwrap().final_stats.clone().unwrap();
        assert_eq!(
            stats.segment_count,
            chunks.iter().filter(|c| !c.text.is_empty()).count()
        );
        let text = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(
            count_run(&text, "ask what you can do for your country"),
            2,
            "{text}"
        );
    }
}

#[tokio::test]
async fn test_streaming_progress_follows_the_audio_consumed() {
    let Ok(model_manager) = ModelManager::new() else {
        println!("⏭ Skipping test - no model directory");
        return;
    };
    let model_path = model_manager.models_dir().join("ggml-base.bin");
    if !model_path.exists() {
        println!("⏭ Skipping test - ggml-base.bin not available");
        return;
    }
    let audio = spoken_twice("jfk.wav").await.unwrap();
    let mut config = TranscriptionConfig::new().with_gpu(false).with_beam_size(1);
    config.model_path = Some(model_path);

    let mut stream = transcribe_file_stream(audio.path(), config).await.unwrap();
    let mut frames = Vec::new();
    while let Some(event) = stream.next().await {
        if let StreamEvent::Progress(progress) = event.unwrap() {
            frames.push(progress);
        }
    }

    let done: Vec<f64> = frames.iter().map(|p| p.frame.transcribed_to).collect();
    assert!(done.windows(2).all(|pair| pair[0] <= pair[1]), "{done:?}");
    assert!(frames.iter().all(|p| p.frame.transcribed_to <= p.received));
    for decoding in frames
        .iter()
        .filter_map(|p| p.frame.decoding.map(|d| (p, d)))
    {
        let (progress, (start, end)) = decoding;
        assert_eq!(start, progress.frame.transcribed_to);
        assert!(end > start && end <= progress.received);
    }
    let last = frames.last().unwrap();
    assert!(last.source_ended);
    assert_eq!(last.frame.decoding, None);
    assert_eq!(last.frame.phase, ProgressPhase::Transcribing);
    assert_eq!(last.frame.transcribed_to, last.received);
    assert!((last.received - 24.0).abs() < 0.1, "{}", last.received);
}

#[tokio::test]
async fn test_regression_consistent_results() {
    if !whisper_fixtures::is_model_available().await {
        println!("⏭ Skipping test - no Whisper model available");
        return;
    }

    let temp_audio = whisper_fixtures::create_short_test_audio().await.unwrap();

    let config = TranscriptionConfig::new()
        .with_gpu(false)
        .with_language("en");

    // Run the same transcription multiple times
    let mut results = Vec::new();
    for _ in 0..3 {
        let result = transcribe_file_sync(temp_audio.path(), Some(config.clone())).await;
        assert!(result.is_ok(), "Regression test run should succeed");
        results.push(result.unwrap());
    }

    // All results should be structurally valid
    for result in &results {
        assert!(validation::validate_sync_result(result));
    }

    // Results should be reasonably consistent (same audio duration)
    let first_duration = results[0].audio_duration;
    for result in &results[1..] {
        assert!(
            (result.audio_duration - first_duration).abs() < 0.1,
            "Audio duration should be consistent across runs"
        );
    }
}
