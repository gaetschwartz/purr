//! Error types for the purr-core library

use thiserror::Error;

/// Main error type for purr operations
#[derive(Error, Debug)]
pub enum WhisperError {
    #[error("Audio processing error: {0}")]
    AudioProcessing(String),

    #[error("Whisper error: {0}")]
    Whisper(whisper_rs::WhisperError),

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

    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Result type alias for purr operations
pub type Result<T> = std::result::Result<T, WhisperError>;

impl From<ffmpeg_next::Error> for WhisperError {
    fn from(err: ffmpeg_next::Error) -> Self {
        WhisperError::FFmpeg(err.to_string())
    }
}

impl PartialEq for WhisperError {
    fn eq(&self, other: &Self) -> bool {
        match self {
            WhisperError::AudioProcessing(msg) => {
                matches!(other, WhisperError::AudioProcessing(o) if msg == o)
            }
            WhisperError::Whisper(msg) => {
                matches!(other, WhisperError::Whisper(o) if msg.to_string() == o.to_string())
            }
            WhisperError::Io(err) => {
                matches!(other, WhisperError::Io(e) if err.to_string() == e.to_string())
            }
            WhisperError::FFmpeg(msg) => {
                matches!(other, WhisperError::FFmpeg(o) if msg == o)
            }
            WhisperError::Configuration(msg) => {
                matches!(other, WhisperError::Configuration(o) if msg == o)
            }
            WhisperError::Transcription(msg) => {
                matches!(other, WhisperError::Transcription(o) if msg == o)
            }
            WhisperError::GpuAcceleration(msg) => {
                matches!(other, WhisperError::GpuAcceleration(o) if msg == o)
            }
            WhisperError::Unknown(msg) => {
                matches!(other, WhisperError::Unknown(o) if msg == o)
            }
        }
    }
}
