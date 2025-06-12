//! Whisper transcription functionality

use crate::audio::AudioData;
use crate::config::TranscriptionConfig;
use crate::error::{Result, WhisperError};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::task;
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

/// Whisper transcriber
pub struct WhisperTranscriber {
    context: WhisperContext,
    config: TranscriptionConfig,
}

impl WhisperTranscriber {
    /// Create a new transcriber with the given configuration
    pub async fn new(config: TranscriptionConfig) -> Result<Self> {
        let config_clone = config.clone();
        
        // Load the model in a blocking task
        let context = task::spawn_blocking(move || {
            Self::load_model(&config_clone)
        }).await
        .map_err(|e| WhisperError::WhisperModel(format!("Task join error: {}", e)))?;
        
        Ok(Self {
            context: context?,
            config,
        })
    }
    
    /// Load the Whisper model
    fn load_model(config: &TranscriptionConfig) -> Result<WhisperContext> {
        // Determine model path
        let model_path = if let Some(path) = &config.model_path {
            path.clone()
        } else {
            // Try to find a default model
            Self::find_default_model()?
        };
        
        // Setup context parameters
        let mut params = WhisperContextParameters::default();
        params.use_gpu(config.use_gpu);
        
        // Load the model
        WhisperContext::new_with_params(&model_path.to_string_lossy(), params)
            .map_err(|e| WhisperError::WhisperModel(format!("Failed to load model: {}", e)))
    }
    
    /// Find a default model file
    fn find_default_model() -> Result<PathBuf> {
        // Common locations for Whisper models
        let possible_paths = [
            "models/ggml-base.en.bin",
            "models/ggml-base.bin", 
            "ggml-base.en.bin",
            "ggml-base.bin",
            "whisper-base.en.bin",
            "whisper-base.bin",
        ];
        
        for path in &possible_paths {
            let path_buf = PathBuf::from(path);
            if path_buf.exists() {
                return Ok(path_buf);
            }
        }
        
        Err(WhisperError::Configuration(
            "No Whisper model found. Please specify a model path or place a model file in the models/ directory".to_string()
        ))
    }
    
    /// Transcribe audio data
    pub async fn transcribe(&self, audio_data: AudioData) -> Result<TranscriptionResult> {
        let context = self.context.clone();
        let config = self.config.clone();
        
        // Run transcription in a blocking task
        task::spawn_blocking(move || {
            Self::transcribe_sync(context, audio_data, config)
        }).await
        .map_err(|e| WhisperError::Transcription(format!("Task join error: {}", e)))?
    }
    
    /// Synchronous transcription implementation
    fn transcribe_sync(
        mut context: WhisperContext, 
        audio_data: AudioData, 
        config: TranscriptionConfig
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
        params.set_print_timestamps(config.output_format.include_timestamps);
        
        // Run transcription
        context
            .full(params, &audio_data.samples)
            .map_err(|e| WhisperError::Transcription(format!("Transcription failed: {}", e)))?;
        
        let processing_time = start_time.elapsed().as_secs_f64();
        
        // Extract results
        let num_segments = context.full_n_segments()
            .map_err(|e| WhisperError::Transcription(format!("Failed to get segment count: {}", e)))?;
        
        let mut segments = Vec::new();
        let mut full_text = String::new();
        
        for i in 0..num_segments {
            let text = context.full_get_segment_text(i)
                .map_err(|e| WhisperError::Transcription(format!("Failed to get segment text: {}", e)))?;
            
            let start = context.full_get_segment_t0(i)
                .map_err(|e| WhisperError::Transcription(format!("Failed to get segment start time: {}", e)))? as f64 / 100.0;
            
            let end = context.full_get_segment_t1(i)
                .map_err(|e| WhisperError::Transcription(format!("Failed to get segment end time: {}", e)))? as f64 / 100.0;
            
            full_text.push_str(&text);
            
            segments.push(TranscriptionSegment {
                text,
                start,
                end,
                confidence: None, // whisper-rs doesn't expose confidence scores yet
                words: None, // Word-level timestamps not implemented yet
            });
        }
        
        // Try to detect language if not specified
        let detected_language = if config.language.is_none() {
            // whisper-rs doesn't expose language detection yet
            None
        } else {
            config.language
        };
        
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
