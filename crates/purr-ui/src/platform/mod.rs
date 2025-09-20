/// Platform abstraction layer for client-only architecture
/// This module provides a unified interface for platform-specific functionality
use bytes::Bytes;
use futures::Stream;
use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    pin::Pin,
    sync::{Arc, LazyLock},
};

/// Re-export the appropriate platform implementation
#[cfg(all(feature = "desktop", not(target_arch = "wasm32")))]
#[path = "default.rs"]
mod platform_impl;

#[cfg(all(not(feature = "desktop"), target_arch = "wasm32"))]
#[path = "wasm.rs"]
mod platform_impl;

#[cfg(all(not(feature = "desktop"), not(target_arch = "wasm32")))]
#[path = "unimplemented.rs"]
mod platform_impl;

/// Platform-specific error types
#[derive(Debug, thiserror::Error, Diagnostic)]
pub enum PlatformError {
    #[error("File processing error: {source}")]
    #[diagnostic(code(platform::file_processing))]
    FileProcessing {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Transcription error: {source}")]
    #[diagnostic(code(platform::transcription))]
    Transcription {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Audio processing error: {source}")]
    #[diagnostic(code(platform::audio_processing))]
    AudioProcessing {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Platform initialization error: {source}")]
    #[diagnostic(code(platform::initialization))]
    Initialization {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("IO error: {source}")]
    #[diagnostic(code(platform::io))]
    Io {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Unsupported operation: {operation}")]
    #[diagnostic(code(platform::unsupported))]
    Unsupported { operation: Cow<'static, str> },

    #[error(transparent)]
    #[diagnostic(transparent)]
    Other(#[from] UnsupportedPlatformError),
}

impl PlatformError {
    /// Create a new PlatformError::FileProcessing
    pub fn file_processing<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        PlatformError::FileProcessing { source: err.into() }
    }

    /// Create a new PlatformError::Transcription
    pub fn transcription<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        PlatformError::Transcription { source: err.into() }
    }

    /// Create a new PlatformError::AudioProcessing
    pub fn audio_processing<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        PlatformError::AudioProcessing { source: err.into() }
    }

    /// Create a new PlatformError::Initialization
    pub fn initialization<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        PlatformError::Initialization { source: err.into() }
    }

    /// Create a new PlatformError::Io
    pub fn io<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        PlatformError::Io { source: err.into() }
    }

    /// Create a new PlatformError::Unsupported
    pub fn unsupported<S>(operation: S) -> Self
    where
        S: Into<Cow<'static, str>>,
    {
        PlatformError::Unsupported {
            operation: operation.into(),
        }
    }

    /// Create a new PlatformError::UnsupportedPlatform
    pub fn unsupported_platform() -> Self {
        PlatformError::Other(UnsupportedPlatformError)
    }
}

#[derive(Debug, Clone, Copy, thiserror::Error, Diagnostic, PartialEq)]
#[diagnostic(code(platform::unsupported_platform))]
pub struct UnsupportedPlatformError;

impl std::fmt::Display for UnsupportedPlatformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(const_str::format!(
            "This operation is not supported on {platform}. ",
            platform = env!("TARGET")
        ))
    }
}

/// Status updates for file processing operations
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ProcessingStatus {
    InProgress { bytes_processed: usize },
    Completed { total_bytes: usize, file_id: String },
    Error { message: String },
}

/// Transcription request parameters
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    /// File data as bytes
    pub file_data: Bytes,
    /// Optional language (auto-detect if None)
    pub language: Option<String>,
    /// Whether to translate to English
    pub translate: bool,
}

/// Transcription status updates
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TranscriptionStatus {
    /// Starting transcription
    Starting,
    /// Processing audio file
    ProcessingAudio,
    /// Transcription in progress with chunk updates
    InProgress {
        chunk_index: usize,
        text: String,
        start_time: f64,
        end_time: f64,
    },
    /// Final result
    Completed {
        processing_time: f64,
        audio_duration: f32,
        word_count: usize,
    },
    /// Error occurred
    Error { message: String },
}

/// Platform trait defining the interface for platform-specific implementations
#[async_trait::async_trait]
pub trait Platform: Send + Sync {
    /// Process a file for transcription (handles temporary storage if needed)
    async fn process_file(
        &self,
        file_data: Bytes,
        file_name: String,
    ) -> Result<String, PlatformError>;

    /// Start transcription of processed file
    async fn transcribe(
        &self,
        file_id: String,
        request: TranscriptionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<TranscriptionStatus, PlatformError>> + Send>>,
        PlatformError,
    >;

    /// Clean up temporary files if any
    async fn cleanup(&self, file_id: &str) -> Result<(), PlatformError>;
}

pub static PLATFORM: LazyLock<Arc<dyn Platform>> =
    LazyLock::new(|| Arc::new(platform_impl::PlatformImpl::new()));
