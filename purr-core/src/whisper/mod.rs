pub mod streaming;
pub mod sync;

use serde::{Deserialize, Serialize};
use std::future::Future;
use tokio::task;
use tracing::info;
use whisper_rs::{WhisperContext, WhisperContextParameters};

use crate::{error::Result, ModelManager, TranscriptionConfig, WhisperError};

pub trait TranscriptionResult {}

/// Transcription statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionStats {
    /// Total processing time in seconds
    pub processing_time: f64,
    
    /// Audio duration in seconds
    pub audio_duration: f32,
    
    /// Real-time factor (audio_duration / processing_time)
    pub real_time_factor: f32,
    
    /// Number of segments produced
    pub segment_count: usize,
    
    /// Average segment length in seconds
    pub avg_segment_length: f32,
    
    /// Total number of words (estimated)
    pub word_count: usize,
    
    /// Words per minute (estimated)
    pub words_per_minute: f32,
}

impl TranscriptionStats {
    pub fn new(processing_time: f64, audio_duration: f32, segment_count: usize, word_count: usize) -> Self {
        let real_time_factor = if processing_time > 0.0 {
            audio_duration as f64 / processing_time
        } else {
            0.0
        } as f32;
        
        let avg_segment_length = if segment_count > 0 {
            audio_duration / segment_count as f32
        } else {
            0.0
        };
        
        let words_per_minute = if audio_duration > 0.0 {
            (word_count as f32 * 60.0) / audio_duration
        } else {
            0.0
        };
        
        Self {
            processing_time,
            audio_duration,
            real_time_factor,
            segment_count,
            avg_segment_length,
            word_count,
            words_per_minute,
        }
    }
}

/// Transcription result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncTranscriptionResult {
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
    
    /// Transcription statistics
    pub stats: TranscriptionStats,
}

impl TranscriptionResult for SyncTranscriptionResult {}

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
    
    /// Final statistics (only present on the very last chunk)
    pub final_stats: Option<TranscriptionStats>,
}

/// Load the Whisper model
pub(crate) async fn load_model(
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

pub trait WhisperTranscriber {
    type TranscriberResult: TranscriptionResult;
    type InputData;

    /// Create a new transcriber with the given configuration
    fn from_config(config: TranscriptionConfig) -> impl Future<Output = Result<Self>>
    where
        Self: Sized;

    /// Transcribe audio data
    fn transcribe(
        self,
        input: Self::InputData,
    ) -> impl Future<Output = Result<Self::TranscriberResult>>;
}
