//! Whisper transcription functionality

use crate::{
    audio::AudioData,
    config::TranscriptionConfig,
    error::{Result, WhisperError},
    whisper::{load_model, SyncTranscriptionResult, TranscriptionSegment, WhisperTranscriber},
    ModelManager,
};
use tracing::warn;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

/// Whisper transcriber
pub struct SyncWhisperTranscriber {
    context: WhisperContext,
    config: TranscriptionConfig,
}

impl WhisperTranscriber for SyncWhisperTranscriber {
    type TranscriberResult = SyncTranscriptionResult;
    type InputData = AudioData;
    /// Create a new transcriber with the given configuration
    async fn from_config(config: TranscriptionConfig) -> Result<Self> {
        // Install logging hooks to redirect whisper.cpp output to Rust logging
        whisper_rs::install_logging_hooks();

        let config_clone = config.clone();

        let model_manager = ModelManager::new()?;

        // Load the model (which may involve async model discovery)
        let context = load_model(&config_clone, &model_manager).await?;

        Ok(Self { context, config })
    }

    /// Transcribe audio data
    async fn transcribe(mut self, audio_data: AudioData) -> Result<SyncTranscriptionResult> {
        let config = self.config.clone();

        // Run transcription synchronously since we can't clone the context
        self.transcribe_sync_internal(audio_data, config)
    }
}

impl SyncWhisperTranscriber {
    /// Synchronous transcription implementation
    fn transcribe_sync_internal(
        &mut self,
        audio_data: AudioData,
        config: TranscriptionConfig,
    ) -> Result<SyncTranscriptionResult> {
        let start_time = std::time::Instant::now();

        // Setup transcription parameters
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Configure parameters
        if let Some(lang) = &config.language {
            params.set_language(Some(lang));
        }

        if let Some(threads) = config.num_threads {
            params.set_n_threads(threads as i32);
        }

        params.set_temperature(config.temperature);

        params.set_print_timestamps(false); // Disable whisper.cpp's internal timestamp printing
        params.set_print_progress(false); // Disable progress output
        params.set_print_special(false); // Disable special token printing
        params.set_print_realtime(false); // Disable real-time printing

        // Create a state for processing
        let mut state = self
            .context
            .create_state()
            .map_err(|e| WhisperError::Transcription(format!("Failed to create state: {}", e)))?;

        // Run transcription using state.full()
        state
            .full(params, &audio_data.samples)
            .map_err(|e| WhisperError::Transcription(format!("Transcription failed: {}", e)))?;

        let processing_time = start_time.elapsed().as_secs_f64();

        // Extract results from state
        let num_segments = state.full_n_segments().map_err(|e| {
            WhisperError::Transcription(format!("Failed to get segment count: {}", e))
        })?;

        let mut segments = Vec::new();
        let mut full_text = String::new();

        for i in 0..num_segments {
            let text = match state.full_get_segment_text(i) {
                Ok(text) => text,
                Err(e) => {
                    warn!(
                        "Failed to get segment text for segment {}: {}. Skipping segment.",
                        i, e
                    );
                    continue; // Skip this segment instead of failing completely
                }
            };

            let start = state.full_get_segment_t0(i).map_err(|e| {
                WhisperError::Transcription(format!("Failed to get segment start time: {}", e))
            })? as f64
                / 100.0;

            let end = state.full_get_segment_t1(i).map_err(|e| {
                WhisperError::Transcription(format!("Failed to get segment end time: {}", e))
            })? as f64
                / 100.0;

            full_text.push_str(&text);

            segments.push(TranscriptionSegment {
                text,
                start,
                end,
                confidence: None, // whisper-rs doesn't expose confidence scores yet
                words: None,      // Word-level timestamps not implemented yet
            });
        }

        // FIXME: Implement language detection
        let detected_language = config.language;

        Ok(SyncTranscriptionResult {
            text: full_text,
            language: detected_language,
            segments,
            processing_time,
            audio_duration: audio_data.duration,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TranscriptionConfig;

    #[tokio::test]
    async fn test_transcription_config() {
        let config = TranscriptionConfig::new()
            .with_language("en")
            .with_gpu(false)
            .with_threads(2);

        assert_eq!(config.language, Some("en".to_string()));
        assert!(!config.use_gpu);
        assert_eq!(config.num_threads, Some(2));
    }

    #[test]
    fn test_transcription_result_serialization() {
        let result = SyncTranscriptionResult {
            text: "Hello world".to_string(),
            language: Some("en".to_string()),
            segments: vec![],
            processing_time: 1.5,
            audio_duration: 3.0,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: SyncTranscriptionResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.text, deserialized.text);
        assert_eq!(result.language, deserialized.language);
    }
}
