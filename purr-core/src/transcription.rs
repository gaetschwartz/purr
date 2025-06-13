//! Whisper transcription functionality

use crate::{
    audio::{AudioData, AudioStream},
    config::TranscriptionConfig,
    error::{Result, WhisperError},
    ModelManager,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::{sync::mpsc, task};
use tracing::{debug, error, info, warn};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Transcription result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    /// The transcribed text
    pub text: String,

    /// Language detected (if auto-detection was used)
    pub language: Option<String>,

    /// Segments with timestamps
    pub segments: Vec<TranscriptionSegment>,

    /// Processing time in seconds
    pub processing_time: f64,

    /// Audio duration in seconds
    pub audio_duration: f32,
}

/// A transcription segment with timestamps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    /// Segment text
    pub text: String,

    /// Start time in seconds
    pub start: f64,

    /// End time in seconds  
    pub end: f64,

    /// Confidence score (if available)
    pub confidence: Option<f32>,

    /// Word-level timestamps (if requested)
    pub words: Option<Vec<WordTimestamp>>,
}

/// Word-level timestamp information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    /// The word
    pub word: String,

    /// Start time in seconds
    pub start: f64,

    /// End time in seconds
    pub end: f64,

    /// Confidence score
    pub confidence: Option<f32>,
}

/// Streaming transcription chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChunk {
    /// Chunk text
    pub text: String,

    /// Start time in seconds
    pub start: f64,

    /// End time in seconds  
    pub end: f64,

    /// Whether this is a final result (true) or partial (false)
    pub is_final: bool,

    /// Chunk index
    pub chunk_index: usize,
}

/// Type alias for streaming result receiver
pub type StreamingReceiver = mpsc::UnboundedReceiver<StreamingChunk>;

/// Whisper transcriber
pub struct WhisperTranscriber {
    context: WhisperContext,
    config: TranscriptionConfig,
}

impl WhisperTranscriber {
    /// Create a new transcriber with the given configuration
    pub async fn new(config: TranscriptionConfig) -> Result<Self> {
        // Install logging hooks to redirect whisper.cpp output to Rust logging
        whisper_rs::install_logging_hooks();

        let config_clone = config.clone();

        let model_manager = ModelManager::new()?;

        // Load the model (which may involve async model discovery)
        let context = Self::load_model(&config_clone, &model_manager).await?;

        Ok(Self { context, config })
    }

    /// Load the Whisper model
    async fn load_model(
        config: &TranscriptionConfig,
        model_manager: &ModelManager,
    ) -> Result<WhisperContext> {
        // Determine model path
        let model_path = if let Some(path) = &config.model_path {
            path.clone()
        } else {
            // Try to find a default model
            model_manager.find_default_model().await?
        };

        if config.verbose {
            info!("Loading Whisper model: {}", model_path.display());
        } else {
            info!("Loading model...");
        }

        // Setup context parameters
        let mut params = WhisperContextParameters::default();
        params.use_gpu(config.use_gpu);

        // Load the model in a blocking task since it's a synchronous operation
        let model_path_str = model_path.to_string_lossy().to_string();
        task::spawn_blocking(move || WhisperContext::new_with_params(&model_path_str, params))
            .await
            .map_err(|e| WhisperError::Unknown(format!("Task join error: {}", e)))?
            .map_err(|e| WhisperError::Whisper(e.to_string()))
    }

    /// Transcribe audio data
    pub async fn transcribe(&mut self, audio_data: AudioData) -> Result<TranscriptionResult> {
        let config = self.config.clone();

        // Run transcription synchronously since we can't clone the context
        Self::transcribe_sync(&mut self.context, audio_data, config)
    }

    /// Start streaming transcription of audio data with simulated real-time output
    pub async fn transcribe_streaming(
        mut self,
        audio_data: AudioData,
    ) -> Result<StreamingReceiver> {
        let (tx, rx) = mpsc::unbounded_channel();

        // Move the transcriber to a background task for processing
        tokio::spawn(async move {
            // Process entire audio first, then stream results with timing
            match self.transcribe(audio_data).await {
                Ok(result) => {
                    // Send segments with simulated real-time delays
                    for (i, segment) in result.segments.iter().enumerate() {
                        if !segment.text.trim().is_empty() {
                            let chunk = StreamingChunk {
                                text: segment.text.clone(),
                                start: segment.start,
                                end: segment.end,
                                is_final: i == result.segments.len() - 1,
                                chunk_index: i,
                            };

                            if tx.send(chunk).is_err() {
                                break; // Receiver dropped
                            }

                            // Add a small delay to simulate real-time streaming
                            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
                        }
                    }
                }
                Err(e) => {
                    error!("Streaming transcription failed: {}", e);
                }
            }
            // Explicitly drop the sender to signal completion
            drop(tx);
        });

        Ok(rx)
    }

    /// True streaming transcription that processes audio chunks as they arrive
    pub async fn transcribe_audio_stream(
        mut self,
        mut audio_stream: AudioStream,
    ) -> Result<StreamingReceiver> {
        let (tx, rx) = mpsc::unbounded_channel();
        let config = self.config.clone();

        // Process chunks in a background task
        tokio::spawn(async move {
            let mut chunk_count = 0;

            while let Some(chunk_result) = audio_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        debug!(
                            "Processing audio chunk {} ({}s duration)",
                            chunk.index, chunk.duration
                        );

                        // Convert chunk to AudioData for whisper processing
                        let audio_data = AudioData {
                            samples: chunk.samples,
                            sample_rate: chunk.sample_rate,
                            duration: chunk.duration,
                        };

                        // Transcribe this chunk
                        match Self::transcribe_sync(&mut self.context, audio_data, config.clone()) {
                            Ok(result) => {
                                // Send each segment from this chunk
                                for segment in result.segments {
                                    if !segment.text.trim().is_empty() {
                                        let streaming_chunk = StreamingChunk {
                                            text: segment.text,
                                            start: chunk.start_time as f64 + segment.start,
                                            end: chunk.start_time as f64 + segment.end,
                                            is_final: chunk.is_final && chunk_count == 0, // Only final if last chunk and last segment
                                            chunk_index: chunk.index,
                                        };

                                        if tx.send(streaming_chunk).is_err() {
                                            warn!("Receiver dropped, stopping transcription");
                                            return;
                                        }
                                    }
                                }

                                chunk_count += 1;

                                // If this was the final chunk, we're done
                                if chunk.is_final {
                                    info!("Completed transcription of {} chunks", chunk_count);
                                    break;
                                }
                            }
                            Err(e) => {
                                error!("Failed to transcribe chunk {}: {}", chunk.index, e);
                                // Continue with next chunk rather than failing completely
                                continue;
                            }
                        }
                    }
                    Err(e) => {
                        error!("Audio streaming error: {}", e);
                        break;
                    }
                }
            }

            // Explicitly drop the sender to signal completion
            drop(tx);
        });

        Ok(rx)
    }

    /// Synchronous transcription implementation
    fn transcribe_sync(
        context: &mut WhisperContext,
        audio_data: AudioData,
        config: TranscriptionConfig,
    ) -> Result<TranscriptionResult> {
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
        let mut state = context
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

        Ok(TranscriptionResult {
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
        let result = TranscriptionResult {
            text: "Hello world".to_string(),
            language: Some("en".to_string()),
            segments: vec![],
            processing_time: 1.5,
            audio_duration: 3.0,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: TranscriptionResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.text, deserialized.text);
        assert_eq!(result.language, deserialized.language);
    }
}
