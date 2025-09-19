//! Error types for the purr-core library

/// Configuration-related errors
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum ConfigurationError {
    #[error("Unknown Whisper model: {model_name}")]
    #[diagnostic(code(whisper::config::unknown_model))]
    UnknownModel { model_name: String },

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

    #[error("Audio processing failed: {reason}")]
    #[diagnostic(code(whisper::audio::processing_failed))]
    ProcessingFailed { reason: String },

    #[error("No audio stream found")]
    #[diagnostic(code(whisper::audio::no_stream))]
    NoAudioStream,

    #[error("Audio format conversion failed")]
    #[diagnostic(code(whisper::audio::format_conversion))]
    FormatConversion,
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
