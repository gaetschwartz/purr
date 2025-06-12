//! Error types for the whisper-ui-core library

use thiserror::Error;

/// Main error type for whisper-ui operations
#[derive(Error, Debug)]
pub enum WhisperError {
    #[error("Audio processing error: {0}")]
    AudioProcessing(String),
    
    #[error("Whisper model error: {0}")]
    WhisperModel(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("FFmpeg error: {0}")]
    FFmpeg(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Transcription error: {0}")]
    Transcription(String),
    
    #[error("GPU acceleration error: {0}")]
    GpuAcceleration(String),
}

/// Result type alias for whisper-ui operations
pub type Result<T> = std::result::Result<T, WhisperError>;

impl From<ffmpeg_next::Error> for WhisperError {
    fn from(err: ffmpeg_next::Error) -> Self {
        WhisperError::FFmpeg(err.to_string())
    }
}
