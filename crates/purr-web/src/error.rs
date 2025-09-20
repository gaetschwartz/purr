//! Error types for the web platform implementation

use purr_common::platform::PlatformError;
use wasm_bindgen::JsValue;

/// Web-specific errors that can occur during platform operations
#[derive(Debug, thiserror::Error)]
pub enum WebError {
    #[error("IndexedDB error: {message}")]
    IndexedDb { message: String },

    #[error("Web API error: {message}")]
    WebApi { message: String },

    #[error("Worker communication error: {message}")]
    Worker { message: String },

    #[error("Model loading error: {message}")]
    ModelLoading { message: String },

    #[error("File processing error: {message}")]
    FileProcessing { message: String },

    #[error("Network error: {message}")]
    Network { message: String },

    #[error("Storage quota exceeded")]
    StorageQuotaExceeded,

    #[error("Browser not supported: {feature}")]
    UnsupportedBrowser { feature: String },

    #[error("JavaScript error: {message}")]
    JavaScript { message: String },

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Worker initialization error: {0}")]
    WorkerInitialization(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Session not found: {0}")]
    SessionNotFound(String),
}

impl WebError {
    /// Create a WebError from a JsValue
    pub fn from_js_value(value: JsValue) -> Self {
        let message = if value.is_string() {
            value.as_string().unwrap_or_else(|| "Unknown JS error".to_string())
        } else {
            format!("{:?}", value)
        };

        WebError::JavaScript { message }
    }

    /// Convert to a PlatformError
    pub fn into_platform_error(self) -> PlatformError {
        match self {
            WebError::FileProcessing { message } => {
                PlatformError::file_processing(Box::new(WebError::FileProcessing { message }))
            }
            WebError::ModelLoading { message } => {
                PlatformError::initialization(Box::new(WebError::ModelLoading { message }))
            }
            WebError::Network { message } => {
                PlatformError::io(Box::new(WebError::Network { message }))
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