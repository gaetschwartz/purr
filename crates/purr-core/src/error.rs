//! Error types for the purr-core library

use std::path::PathBuf;

// ========================================
// Dedicated Error Types for Transparent Wrapper Pattern
// ========================================

/// Error for unknown Whisper model names
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
#[error("Unknown Whisper model: {model_name}")]
#[diagnostic(code(whisper::config::unknown_model))]
pub struct UnknownModelError {
    pub model_name: String,
}

impl UnknownModelError {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
        }
    }
}

/// Error for invalid audio parameters
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
#[error("Invalid audio parameters: {details}")]
#[diagnostic(code(whisper::audio::invalid_parameters))]
pub struct InvalidParametersError {
    pub details: String,
}

impl InvalidParametersError {
    pub fn new(details: impl Into<String>) -> Self {
        Self {
            details: details.into(),
        }
    }
}

/// Error for unsupported audio formats
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
#[error("Unsupported audio format: {format}")]
#[diagnostic(code(whisper::audio::unsupported_format))]
pub struct UnsupportedFormatError {
    pub format: String,
}

impl UnsupportedFormatError {
    pub fn new(format: impl Into<String>) -> Self {
        Self {
            format: format.into(),
        }
    }
}

/// Configuration-related errors
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum ConfigurationError {
    #[error(transparent)]
    #[diagnostic(transparent)]
    UnknownModel(UnknownModelError),

    #[error("Failed to get XDG directories")]
    #[diagnostic(code(whisper::config::xdg_dirs))]
    XdgDirectories,

    #[error("Model download failed: {source}")]
    #[diagnostic(code(whisper::config::download_failed))]
    DownloadFailed {
        #[source]
        source: reqwest::Error,
    },

    #[error("Failed to download model {model_name}: HTTP {status_code}")]
    #[diagnostic(code(whisper::config::download_http_error))]
    DownloadHttpError {
        model_name: String,
        status_code: u16,
    },

    #[error("Download stream error: {source}")]
    #[diagnostic(code(whisper::config::download_stream_error))]
    DownloadStreamError {
        #[source]
        source: reqwest::Error,
    },

    #[error("No Whisper model found")]
    #[diagnostic(code(whisper::config::no_model_found))]
    NoModelFound,
}

impl ConfigurationError {
    /// Create a new unknown model error
    pub fn unknown_model(model_name: impl Into<String>) -> Self {
        Self::UnknownModel(UnknownModelError::new(model_name))
    }
}

/// Transcription-related errors
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum TranscriptionError {
    #[error("Failed to create Whisper state: {source}")]
    #[diagnostic(code(whisper::transcription::state_creation))]
    StateCreation {
        #[source]
        source: whisper_rs::WhisperError,
    },

    #[error("Transcription failed: {source}")]
    #[diagnostic(code(whisper::transcription::failed))]
    Failed {
        #[source]
        source: whisper_rs::WhisperError,
    },
}

/// Audio processing related errors
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum AudioProcessingError {
    #[error("Task join error: {source}")]
    #[diagnostic(code(whisper::audio::task_join))]
    TaskJoin {
        #[source]
        source: tokio::task::JoinError,
    },

    #[error("Audio processing failed: {operation}")]
    #[diagnostic(code(whisper::audio::processing_failed))]
    ProcessingFailed {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Audio decoding failed: {codec} format not supported")]
    #[diagnostic(code(whisper::audio::decode_failed))]
    DecodeFailed {
        codec: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Audio resampling failed from {source_rate}Hz to {target_rate}Hz")]
    #[diagnostic(code(whisper::audio::resample_failed))]
    ResampleFailed {
        source_rate: u32,
        target_rate: u32,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Audio file too large: {file_size} bytes exceeds limit of {max_size} bytes")]
    #[diagnostic(code(whisper::audio::file_too_large))]
    FileTooLarge {
        file_size: u64,
        max_size: u64,
    },

    #[error(transparent)]
    #[diagnostic(transparent)]
    InvalidParameters(InvalidParametersError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    UnsupportedFormat(UnsupportedFormatError),

    #[error("Failed to read audio file {file}: {source}")]
    #[diagnostic(code(whisper::audio::read_failed))]
    ReadFailed {
        file: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("No audio stream found")]
    #[diagnostic(code(whisper::audio::no_stream))]
    NoAudioStream,

    #[error("Audio format conversion failed")]
    #[diagnostic(code(whisper::audio::format_conversion))]
    FormatConversion,
}

impl AudioProcessingError {
    /// Create a new processing failed error with structured information
    pub fn processing_failed<E>(operation: impl Into<String>, source: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::ProcessingFailed {
            operation: operation.into(),
            source: source.into(),
        }
    }

    /// Create a new decode failed error
    pub fn decode_failed<E>(codec: impl Into<String>, source: Option<E>) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::DecodeFailed {
            codec: codec.into(),
            source: source.map(Into::into),
        }
    }

    /// Create a new resample failed error
    pub fn resample_failed<E>(source_rate: u32, target_rate: u32, source: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::ResampleFailed {
            source_rate,
            target_rate,
            source: source.into(),
        }
    }

    /// Create a new file too large error
    pub fn file_too_large(file_size: u64, max_size: u64) -> Self {
        Self::FileTooLarge {
            file_size,
            max_size,
        }
    }

    /// Create a new invalid parameters error
    pub fn invalid_parameters(details: impl Into<String>) -> Self {
        Self::InvalidParameters(InvalidParametersError::new(details))
    }

    /// Create a new unsupported format error
    pub fn unsupported_format(format: impl Into<String>) -> Self {
        Self::UnsupportedFormat(UnsupportedFormatError::new(format))
    }
}

/// Main error type for purr operations
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum WhisperError {
    #[error("Audio processing error: {source}")]
    #[diagnostic(code(whisper::audio::processing))]
    AudioProcessing {
        #[from]
        source: AudioProcessingError,
    },

    #[error("Whisper model error: {source}")]
    #[diagnostic(code(whisper::model::error))]
    Whisper {
        #[from]
        source: whisper_rs::WhisperError,
    },

    #[error("IO operation failed: {source}")]
    #[diagnostic(code(whisper::io::error))]
    Io {
        #[from]
        source: std::io::Error,
    },

    #[error("FFmpeg processing error: {source}")]
    #[diagnostic(code(whisper::ffmpeg::error))]
    FFmpeg {
        #[from]
        source: ffmpeg_next::Error,
    },

    #[error("JSON serialization/deserialization error: {source}")]
    #[diagnostic(code(whisper::json::error))]
    Json {
        #[from]
        source: serde_json::Error,
    },

    #[error("HTTP request error: {source}")]
    #[diagnostic(code(whisper::http::error))]
    Http {
        #[from]
        source: reqwest::Error,
    },

    #[error("Configuration error: {source}")]
    #[diagnostic(code(whisper::config::error))]
    Configuration {
        #[from]
        source: ConfigurationError,
    },

    #[error("Transcription failed: {source}")]
    #[diagnostic(code(whisper::transcription::error))]
    Transcription {
        #[from]
        source: TranscriptionError,
    },

    #[error("Tracing/logging error: {source}")]
    #[diagnostic(code(whisper::tracing::error))]
    Tracing {
        #[from]
        source: tracing_subscriber::util::TryInitError,
    },

    #[error("Environment parsing error: {source}")]
    #[diagnostic(code(whisper::env::from_error))]
    EnvFrom {
        #[from]
        source: tracing_subscriber::filter::FromEnvError,
    },

    #[error("Directive parsing error: {source}")]
    #[diagnostic(code(whisper::env::parse_error))]
    EnvParse {
        #[from]
        source: tracing_subscriber::filter::ParseError,
    },

    #[error("Task join error: {source}")]
    #[diagnostic(code(whisper::task_join::error))]
    TaskJoin {
        #[from]
        source: tokio::task::JoinError,
    },
}

/// Result type alias for purr operations
pub type Result<T> = std::result::Result<T, WhisperError>;
