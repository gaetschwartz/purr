//! Error types for the purr-core library

use crate::model::digest::Sha256Digest;
use purr_common::model::{engine::EngineId, format::ModelFormat};
use std::path::PathBuf;

use crate::{
    accel::report::{AccelAttempt, AttemptOutcome},
    engine::error::{DecodeError, DetectError, OpenError},
};

// ========================================
// Dedicated Error Types for Transparent Wrapper Pattern
// ========================================

/// Error for unknown Whisper model names
#[oopsie::oopsie]
#[oopsie(
    "Unknown Whisper model: {model_name} (see 'models list --available')",
    code = "whisper::config::unknown_model"
)]
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
#[oopsie::oopsie]
#[derive(Clone, PartialEq, Eq)]
pub enum ManifestError {
    #[oopsie(
        "Failed to parse the model manifest: {message}",
        code = "whisper::manifest::parse"
    )]
    Parse { message: String },

    #[oopsie(
        "Unsupported model manifest schema {found} (supported: {supported})",
        code = "whisper::manifest::unsupported_schema"
    )]
    UnsupportedSchema { found: u32, supported: u32 },

    #[oopsie(
        "Duplicate model id in manifest: {id}",
        code = "whisper::manifest::duplicate_model_id"
    )]
    DuplicateModelId { id: String },

    #[oopsie(
        "Duplicate source id in manifest: {id}",
        code = "whisper::manifest::duplicate_source_id"
    )]
    DuplicateSourceId { id: String },

    #[oopsie(
        "Duplicate model filename in manifest: {filename}",
        code = "whisper::manifest::duplicate_filename"
    )]
    DuplicateFilename { filename: String },

    #[oopsie(
        "Model {model_id} references unknown source {source_id}",
        code = "whisper::manifest::unknown_source"
    )]
    UnknownSource { model_id: String, source_id: String },

    #[oopsie(
        "Model {model_id} is run by {engine}, which does not load {format} files",
        code = "whisper::manifest::unsupported_format"
    )]
    UnsupportedFormat {
        model_id: String,
        engine: EngineId,
        format: ModelFormat,
    },

    #[oopsie(
        "Model {model_id} refers to unknown profile {profile}",
        code = "whisper::manifest::unknown_profile"
    )]
    UnknownProfile { model_id: String, profile: String },

    #[oopsie(
        "Source {source_id} has an invalid revision: {revision}",
        code = "whisper::manifest::invalid_revision"
    )]
    InvalidRevision { source_id: String, revision: String },

    #[oopsie(
        "Model {model_id} has a size of zero bytes",
        code = "whisper::manifest::zero_size"
    )]
    ZeroSize { model_id: String },

    #[oopsie(
        "Model {model_id} has no SHA-256 digest",
        code = "whisper::manifest::missing_digest"
    )]
    MissingDigest { model_id: String },

    #[oopsie(
        "The manifest lists no preferred models",
        code = "whisper::manifest::no_preferred_model"
    )]
    NoPreferredModel,

    #[oopsie(
        "Preferred model {id} is not in the manifest",
        code = "whisper::manifest::unknown_preferred_model"
    )]
    UnknownPreferredModel { id: String },
}

/// Error for a model file that does not match the manifest
#[oopsie::oopsie]
#[derive(Clone, PartialEq, Eq)]
pub enum ModelIntegrityError {
    #[oopsie(
        "Model {model_id} has {actual} bytes, expected {expected}",
        code = "whisper::model::size_mismatch"
    )]
    SizeMismatch {
        model_id: String,
        expected: u64,
        actual: u64,
    },

    #[oopsie(
        "Model {model_id} has SHA-256 {actual}, expected {expected}",
        code = "whisper::model::checksum_mismatch"
    )]
    ChecksumMismatch {
        model_id: String,
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
}

/// Error for unsupported audio formats
#[oopsie::oopsie]
#[oopsie(
    "Unsupported audio format: {format}",
    code = "whisper::audio::unsupported_format"
)]
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
#[oopsie::oopsie]
pub enum ConfigurationError {
    #[oopsie(transparent)]
    UnknownModel { source: UnknownModelError },

    #[oopsie(transparent)]
    Manifest { source: ManifestError },

    #[oopsie(transparent)]
    ModelIntegrity { source: ModelIntegrityError },

    #[oopsie("Failed to get XDG directories", code = "whisper::config::xdg_dirs")]
    XdgDirectories,

    #[oopsie("Model download failed", code = "whisper::config::download_failed")]
    DownloadFailed { source: reqwest::Error },

    #[oopsie(
        "Failed to download model {model_name}: HTTP {status_code}",
        code = "whisper::config::download_http_error"
    )]
    DownloadHttpError {
        model_name: String,
        status_code: u16,
    },

    #[oopsie(
        "Download stream error",
        code = "whisper::config::download_stream_error"
    )]
    DownloadStreamError { source: reqwest::Error },

    #[oopsie(
        "No model found that this build can run",
        code = "whisper::config::no_model_found"
    )]
    NoModelFound,

    #[oopsie("{} is not a model file Purr reads (GGUF or ggml)", path.display(), code = "whisper::config::unknown_model_file")]
    UnknownModelFile { path: PathBuf },
}

impl ConfigurationError {
    /// Create a new unknown model error
    pub fn unknown_model(model_name: impl Into<String>) -> Self {
        Self::UnknownModel {
            source: UnknownModelError::new(model_name),
        }
    }
}

/// Transcription-related errors
#[oopsie::oopsie]
pub enum TranscriptionError {
    #[oopsie("Transcription failed", code = "whisper::transcription::failed")]
    Decode { source: DecodeError },

    #[oopsie(
        "Voice activity detection failed",
        code = "whisper::transcription::speech_detection"
    )]
    Detect { source: DetectError },

    #[oopsie("Failed to load the model", code = "whisper::transcription::open")]
    Open { source: OpenError },

    #[oopsie(
        "The model does not know the language `{code}`",
        code = "whisper::transcription::unknown_language"
    )]
    UnknownLanguage { code: String },

    #[oopsie("Failed to load the voice activity model {}", path.display(), code = "whisper::transcription::vad_model")]
    VadModel {
        path: std::path::PathBuf,
        source: OpenError,
    },

    #[oopsie(
        "This build has no {engine} engine",
        code = "whisper::transcription::engine_missing"
    )]
    EngineMissing { engine: EngineId },

    #[oopsie("The {} cannot be used: {reason}", engine.label(), code = "whisper::transcription::engine_unavailable")]
    EngineUnavailable {
        engine: EngineId,
        reason: purr_common::accel::EngineUnavailable,
    },
}

/// Acceleration backend selection errors
#[oopsie::oopsie]
pub enum AccelerationError {
    #[oopsie(
        "No device could load the model: {}",
        describe_attempts(attempts),
        code = "whisper::accel::exhausted"
    )]
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
#[oopsie::oopsie]
pub enum AudioProcessingError {
    #[oopsie("Task join error", code = "whisper::audio::task_join")]
    TaskJoin { source: tokio::task::JoinError },

    #[oopsie(
        "Audio processing failed: {operation}",
        code = "whisper::audio::processing_failed"
    )]
    ProcessingFailed {
        operation: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[oopsie("Failed to read audio file {}", file.display(), code = "whisper::audio::read_failed")]
    ReadFailed {
        file: PathBuf,
        source: std::io::Error,
    },

    #[oopsie("No audio stream found", code = "whisper::audio::no_stream")]
    NoAudioStream,

    #[oopsie(
        "No audio could be decoded from the input, it may be corrupted or unsupported",
        code = "whisper::audio::no_data"
    )]
    NoAudioData,

    #[oopsie(
        "Audio format conversion failed",
        code = "whisper::audio::format_conversion"
    )]
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
#[oopsie::oopsie]
pub enum Error {
    #[oopsie(
        "Audio processing error",
        code = "whisper::audio::processing",
        transparent
    )]
    AudioProcessing { source: AudioProcessingError },

    #[oopsie(
        "Failed to create the models directory {}",
        path.display(),
        code = "whisper::model::create_dir_failed"
    )]
    CreateModelsDir {
        path: PathBuf,
        source: std::io::Error,
    },

    #[oopsie(
        "Failed to open the model file {}",
        path.display(),
        code = "whisper::model::open_failed"
    )]
    OpenModelFile {
        path: PathBuf,
        source: std::io::Error,
    },

    #[oopsie(
        "Failed to read the model file {}",
        path.display(),
        code = "whisper::model::read_failed"
    )]
    ReadModelFile {
        path: PathBuf,
        source: std::io::Error,
    },

    #[oopsie(
        "Failed to inspect the model file {}",
        path.display(),
        code = "whisper::model::inspect_failed"
    )]
    InspectModelFile {
        path: PathBuf,
        source: std::io::Error,
    },

    #[oopsie(
        "Failed to delete the model file {}",
        path.display(),
        code = "whisper::model::delete_failed"
    )]
    DeleteModelFile {
        path: PathBuf,
        source: std::io::Error,
    },

    #[oopsie(
        "Failed to write the download to {}",
        path.display(),
        code = "whisper::download::write_failed"
    )]
    WriteDownload {
        path: PathBuf,
        source: std::io::Error,
    },

    #[oopsie(
        "Failed to move the download into place at {}",
        path.display(),
        code = "whisper::download::move_failed"
    )]
    MoveDownload {
        path: PathBuf,
        source: std::io::Error,
    },

    #[oopsie("Failed to initialise FFmpeg", code = "whisper::ffmpeg::init")]
    InitFfmpeg { source: ffmpeg_next::Error },

    #[oopsie(
        "Failed to open the audio input {}",
        path.display(),
        code = "whisper::ffmpeg::open_input"
    )]
    OpenAudioInput {
        path: PathBuf,
        source: ffmpeg_next::Error,
    },

    #[oopsie(
        "Failed to open the streamed audio input",
        code = "whisper::ffmpeg::open_stream_input"
    )]
    OpenStreamInput { source: ffmpeg_next::Error },

    #[oopsie(
        "Failed to create the audio decoder",
        code = "whisper::ffmpeg::create_decoder"
    )]
    CreateAudioDecoder { source: ffmpeg_next::Error },

    #[oopsie(
        "Failed to decode audio stream {stream_index}",
        code = "whisper::ffmpeg::decode"
    )]
    DecodeAudioPacket {
        stream_index: usize,
        source: ffmpeg_next::Error,
    },

    #[oopsie(
        "Failed to create the audio resampler",
        code = "whisper::ffmpeg::create_resampler"
    )]
    CreateResampler { source: ffmpeg_next::Error },

    #[oopsie(
        "Failed to flush the audio resampler",
        code = "whisper::ffmpeg::flush_resampler"
    )]
    FlushResampler { source: ffmpeg_next::Error },

    #[oopsie(
        "JSON serialization/deserialization error",
        code = "whisper::json::error"
    )]
    Json { source: serde_json::Error },

    #[oopsie("Invalid audio URL", code = "whisper::input::invalid_url")]
    InvalidUrl { source: reqwest::Error },

    #[oopsie("Configuration error", code = "whisper::config::error", transparent)]
    Configuration { source: ConfigurationError },

    #[oopsie(
        "Transcription failed",
        code = "whisper::transcription::error",
        transparent
    )]
    Transcription { source: TranscriptionError },

    #[oopsie("Acceleration error", code = "whisper::accel::error", transparent)]
    Acceleration { source: AccelerationError },

    #[oopsie("Tracing/logging error", code = "whisper::tracing::error")]
    Tracing {
        source: tracing_subscriber::util::TryInitError,
    },

    #[oopsie("Environment parsing error", code = "whisper::env::from_error")]
    EnvFrom {
        source: tracing_subscriber::filter::FromEnvError,
    },

    #[oopsie("Directive parsing error", code = "whisper::env::parse_error")]
    EnvParse {
        source: tracing_subscriber::filter::ParseError,
    },

    #[oopsie("The {task} task did not finish", code = "whisper::task_join::error")]
    TaskJoin {
        task: &'static str,
        source: tokio::task::JoinError,
    },

    #[oopsie("Network error: {message}", code = "whisper::network::error")]
    NetworkError { message: String },

    #[oopsie("Invalid input: {message}", code = "whisper::input::error")]
    InvalidInput { message: String },
}

macro_rules! into_whisper_error_via_configuration {
    ($($error:ty),+ $(,)?) => {$(
        impl From<$error> for Error {
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
pub type Result<T> = std::result::Result<T, Error>;
