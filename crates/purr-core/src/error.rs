//! Error types for the purr-core library

use crate::model::digest::Sha256Digest;
use std::path::PathBuf;

use crate::accel::report::{AccelAttempt, AttemptOutcome};

// ========================================
// Dedicated Error Types for Transparent Wrapper Pattern
// ========================================

/// Error for unknown Whisper model names
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
#[error("Unknown Whisper model: {model_name} (see 'models list --available')")]
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

/// Error for an invalid or inconsistent model manifest
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, miette::Diagnostic)]
pub enum ManifestError {
    #[error("Failed to parse the model manifest: {message}")]
    #[diagnostic(code(whisper::manifest::parse))]
    Parse { message: String },

    #[error("Unsupported model manifest schema {found} (supported: {supported})")]
    #[diagnostic(code(whisper::manifest::unsupported_schema))]
    UnsupportedSchema { found: u32, supported: u32 },

    #[error("Duplicate model id in manifest: {id}")]
    #[diagnostic(code(whisper::manifest::duplicate_model_id))]
    DuplicateModelId { id: String },

    #[error("Duplicate source id in manifest: {id}")]
    #[diagnostic(code(whisper::manifest::duplicate_source_id))]
    DuplicateSourceId { id: String },

    #[error("Duplicate model filename in manifest: {filename}")]
    #[diagnostic(code(whisper::manifest::duplicate_filename))]
    DuplicateFilename { filename: String },

    #[error("Model {model_id} references unknown source {source_id}")]
    #[diagnostic(code(whisper::manifest::unknown_source))]
    UnknownSource { model_id: String, source_id: String },

    #[error("Source {source_id} has an invalid revision: {revision}")]
    #[diagnostic(code(whisper::manifest::invalid_revision))]
    InvalidRevision { source_id: String, revision: String },

    #[error("Model {model_id} has a size of zero bytes")]
    #[diagnostic(code(whisper::manifest::zero_size))]
    ZeroSize { model_id: String },

    #[error("Model {model_id} has no SHA-256 digest")]
    #[diagnostic(code(whisper::manifest::missing_digest))]
    MissingDigest { model_id: String },

    #[error("The manifest lists no preferred models")]
    #[diagnostic(code(whisper::manifest::no_preferred_model))]
    NoPreferredModel,

    #[error("Preferred model {id} is not in the manifest")]
    #[diagnostic(code(whisper::manifest::unknown_preferred_model))]
    UnknownPreferredModel { id: String },
}

/// Error for a model file that does not match the manifest
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, miette::Diagnostic)]
pub enum ModelIntegrityError {
    #[error("Model {model_id} has {actual} bytes, expected {expected}")]
    #[diagnostic(code(whisper::model::size_mismatch))]
    SizeMismatch {
        model_id: String,
        expected: u64,
        actual: u64,
    },

    #[error("Model {model_id} has SHA-256 {actual}, expected {expected}")]
    #[diagnostic(code(whisper::model::checksum_mismatch))]
    ChecksumMismatch {
        model_id: String,
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
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
    UnknownModel(#[from] UnknownModelError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Manifest(#[from] ManifestError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    ModelIntegrity(#[from] ModelIntegrityError),

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

    #[error("Whisper does not know the language `{code}`")]
    #[diagnostic(code(whisper::transcription::unknown_language))]
    UnknownLanguage { code: String },

    #[error("Whisper detected a language id it has no code for: {id}")]
    #[diagnostic(code(whisper::transcription::unknown_language_id))]
    UnknownLanguageId { id: i32 },

    #[error("Failed to load the voice activity model {}: {source}", path.display())]
    #[diagnostic(code(whisper::transcription::vad_model))]
    VadModel {
        path: std::path::PathBuf,
        #[source]
        source: whisper_rs::WhisperError,
    },
}

/// Acceleration backend selection errors
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum AccelerationError {
    #[error("No device could load the model: {}", describe_attempts(attempts))]
    #[diagnostic(code(whisper::accel::exhausted))]
    Exhausted { attempts: Vec<AccelAttempt> },
}

fn describe_attempts(attempts: &[AccelAttempt]) -> String {
    attempts
        .iter()
        .filter(|attempt| matches!(attempt.outcome, AttemptOutcome::Failed(_)))
        .map(|attempt| {
            if attempt.key.is_cpu() {
                format!("{}: {}", attempt.backend.pretty_name(), attempt.outcome)
            } else {
                format!(
                    "{} {}: {}",
                    attempt.backend.pretty_name(),
                    attempt.key.name,
                    attempt.outcome
                )
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
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

    #[error("Acceleration error: {source}")]
    #[diagnostic(code(whisper::accel::error))]
    Acceleration {
        #[from]
        source: AccelerationError,
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

    #[error("Network error: {message}")]
    #[diagnostic(code(whisper::network::error))]
    NetworkError { message: String },

    #[error("Invalid input: {message}")]
    #[diagnostic(code(whisper::input::error))]
    InvalidInput { message: String },
}

macro_rules! into_whisper_error_via_configuration {
    ($($error:ty),+ $(,)?) => {$(
        impl From<$error> for WhisperError {
            fn from(source: $error) -> Self {
                Self::Configuration {
                    source: ConfigurationError::from(source),
                }
            }
        }
    )+};
}

into_whisper_error_via_configuration!(UnknownModelError, ManifestError, ModelIntegrityError);

/// Result type alias for purr operations
pub type Result<T> = std::result::Result<T, WhisperError>;
