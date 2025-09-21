//! Error types for the web platform implementation

use purr_common::platform::PlatformError;
use wasm_bindgen::JsValue;

// ========================================
// Dedicated Error Types for Transparent Wrapper Pattern
// ========================================

/// Error for session not found scenarios
#[derive(Debug, thiserror::Error)]
#[error("Session not found: {session_id}")]
pub struct SessionNotFoundError {
    pub session_id: String,
}

impl SessionNotFoundError {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
        }
    }
}

/// Web-specific errors that can occur during platform operations
#[derive(Debug, thiserror::Error)]
pub enum WebError {
    #[error("IndexedDB operation failed: {operation}")]
    IndexedDb {
        operation: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Web API error: {api} call failed")]
    WebApi {
        api: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Worker communication error: {worker_type} worker failed")]
    Worker {
        worker_type: String,
        operation: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Model loading error: {model_id} failed to load")]
    ModelLoading {
        model_id: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("File processing error: {operation} failed for {file_type}")]
    FileProcessing {
        operation: String,
        file_type: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Network error: {method} {url} failed")]
    Network {
        method: String,
        url: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Storage quota exceeded")]
    StorageQuotaExceeded,

    #[error("Browser feature not supported: {feature} (requires {minimum_version})")]
    UnsupportedBrowser {
        feature: String,
        minimum_version: String,
        current_browser: String,
    },

    #[error("JavaScript error: {context}")]
    JavaScript {
        context: String,
        js_error: String,
    },

    #[error("Model not found: {model_id} in {storage_type} storage")]
    ModelNotFound {
        model_id: String,
        storage_type: String,
    },

    #[error("Network error: {status_code} {status_text}")]
    NetworkError {
        status_code: u16,
        status_text: String,
        url: String,
    },

    #[error("Storage error: {operation} failed in {storage_type}")]
    StorageError {
        operation: String,
        storage_type: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Worker initialization error: {worker_type} failed to start")]
    WorkerInitialization {
        worker_type: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Serialization error: {data_type} serialization failed")]
    Serialization {
        data_type: String,
        operation: String, // "serialize" or "deserialize"
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error(transparent)]
    SessionNotFound(SessionNotFoundError),

    #[error("Audio processing error: {operation} failed on {format} data")]
    AudioProcessing {
        operation: String,
        format: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("WebGPU error: {operation} failed on {device_type}")]
    WebGpu {
        operation: String,
        device_type: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Audio format not supported: {format} (supported: {supported_formats})")]
    UnsupportedAudioFormat {
        format: String,
        supported_formats: String,
    },

    #[error("Audio file too large: {file_size} bytes exceeds limit of {max_size} bytes")]
    AudioFileTooLarge {
        file_size: u64,
        max_size: u64,
    },

    #[error("Web Audio API error: {operation} failed")]
    WebAudioApi {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl WebError {
    /// Create a new session not found error
    pub fn session_not_found(session_id: impl Into<String>) -> Self {
        Self::SessionNotFound(SessionNotFoundError::new(session_id))
    }

    /// Create a new model not found error
    pub fn model_not_found(model_id: impl Into<String>, storage_type: impl Into<String>) -> Self {
        Self::ModelNotFound {
            model_id: model_id.into(),
            storage_type: storage_type.into(),
        }
    }

    /// Create a new network error
    pub fn network_error(status_code: u16, status_text: impl Into<String>, url: impl Into<String>) -> Self {
        Self::NetworkError {
            status_code,
            status_text: status_text.into(),
            url: url.into(),
        }
    }

    /// Create a new storage error
    pub fn storage_error<E>(operation: impl Into<String>, storage_type: impl Into<String>, source: Option<E>) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::StorageError {
            operation: operation.into(),
            storage_type: storage_type.into(),
            source: source.map(Into::into),
        }
    }

    /// Create a new audio processing error
    pub fn audio_processing<E>(operation: impl Into<String>, format: impl Into<String>, source: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::AudioProcessing {
            operation: operation.into(),
            format: format.into(),
            source: source.into(),
        }
    }



    /// Create a WebError from a JsValue
    pub fn from_js_value(value: JsValue) -> Self {
        let message = if value.is_string() {
            value.as_string().unwrap_or_else(|| "Unknown JS error".to_string())
        } else {
            format!("{:?}", value)
        };

        WebError::JavaScript {
            context: "JavaScript execution".to_string(),
            js_error: message,
        }
    }

    /// Convert to a PlatformError
    pub fn into_platform_error(self) -> PlatformError {
        match self {
            WebError::FileProcessing { source, .. } => {
                PlatformError::file_processing(source)
            }
            WebError::ModelLoading { source, .. } => {
                PlatformError::initialization(source)
            }
            WebError::Network { source, .. } => {
                PlatformError::io(source)
            }
            WebError::AudioProcessing { source, .. } => {
                PlatformError::audio_processing(source)
            }
            WebError::UnsupportedAudioFormat { .. } => {
                PlatformError::audio_processing(Box::new(self))
            }
            WebError::AudioFileTooLarge { .. } => {
                PlatformError::audio_processing(Box::new(self))
            }
            WebError::WebAudioApi { source, .. } => {
                PlatformError::audio_processing(source)
            }
            WebError::WebGpu { source, .. } => {
                PlatformError::initialization(source)
            }
            _ => PlatformError::io(Box::new(self)),
        }
    }
}

impl From<JsValue> for WebError {
    fn from(value: JsValue) -> Self {
        WebError::from_js_value(value)
    }
}

impl From<WebError> for JsValue {
    fn from(error: WebError) -> Self {
        JsValue::from_str(&error.to_string())
    }
}

impl From<WebError> for PlatformError {
    fn from(error: WebError) -> Self {
        error.into_platform_error()
    }
}

/// Result type for web operations
pub type WebResult<T> = Result<T, WebError>;