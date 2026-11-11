//! Error types for the web platform implementation

use purr_common::platform::PlatformError;
use purr_common::web_model::error::{UnknownWebModelError, WebManifestError};
use wasm_bindgen::JsValue;

/// Helper function to format `JsValue` errors into strings while preserving information
pub fn format_js_error(js_error: JsValue) -> String {
    if js_error.is_string() {
        js_error
            .as_string()
            .unwrap_or_else(|| "Unknown JS error".to_owned())
    } else if js_error.is_object() {
        // Try to get error message, name, and stack if available
        let obj = js_sys::Object::from(js_error);
        let mut parts = Vec::new();

        if let Ok(name) = js_sys::Reflect::get(&obj, &"name".into())
            && let Some(name_str) = name.as_string()
        {
            parts.push(format!("name: {name_str}"));
        }

        if let Ok(message) = js_sys::Reflect::get(&obj, &"message".into())
            && let Some(message_str) = message.as_string()
        {
            parts.push(format!("message: {message_str}"));
        }

        if let Ok(stack) = js_sys::Reflect::get(&obj, &"stack".into())
            && let Some(stack_str) = stack.as_string()
        {
            parts.push(format!("stack: {stack_str}"));
        }

        if parts.is_empty() {
            format!("{obj:?}")
        } else {
            parts.join(", ")
        }
    } else {
        format!("{js_error:?}")
    }
}

// ========================================
// Specialized Error Enums for Domain-Specific Error Handling
// ========================================

/// Failures of the transcription worker
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum WorkerError {
    #[error("The transcription worker was shut down before the request finished")]
    #[diagnostic(code(purr::web::worker::closed))]
    Closed,

    #[error("{message}")]
    #[diagnostic(code(purr::web::worker::request_failed))]
    RequestFailed {
        kind: crate::protocol::FailureKind,
        message: String,
    },
}

/// Storage-specific errors for `IndexedDB` and other storage operations
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum StorageError {
    #[error("IndexedDB availability check failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::indexeddb_unavailable))]
    IndexedDbUnavailable { js_error: String },

    #[error("Database open failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::database_open_failed))]
    DatabaseOpenFailed { js_error: String },

    #[error("Database connection failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::database_connection_failed))]
    DatabaseConnectionFailed { js_error: String },

    #[error("Database cast failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::database_cast_failed))]
    DatabaseCastFailed { js_error: String },

    #[error("Transaction creation failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::transaction_creation_failed))]
    TransactionCreationFailed { js_error: String },

    #[error("Object store access failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::object_store_access_failed))]
    ObjectStoreAccessFailed { js_error: String },

    #[error("File store operation failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::file_store_failed))]
    FileStoreFailed { js_error: String },

    #[error("File storage operation failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::file_storage_operation_failed))]
    FileStorageOperationFailed { js_error: String },

    #[error("Metadata transaction creation failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::metadata_transaction_failed))]
    MetadataTransactionFailed { js_error: String },

    #[error("Metadata store access failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::metadata_store_access_failed))]
    MetadataStoreAccessFailed { js_error: String },

    #[error("Metadata serialization failed")]
    #[diagnostic(code(purr::web::storage::metadata_serialization_failed))]
    MetadataSerializationFailed,

    #[error("Metadata store operation failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::metadata_store_failed))]
    MetadataStoreFailed { js_error: String },

    #[error("Metadata storage operation failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::metadata_storage_operation_failed))]
    MetadataStorageOperationFailed { js_error: String },

    #[error("Get request failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::get_request_failed))]
    GetRequestFailed { js_error: String },

    #[error("Get operation failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::get_operation_failed))]
    GetOperationFailed { js_error: String },

    #[error("Data format invalid")]
    #[diagnostic(code(purr::web::storage::data_format_invalid))]
    DataFormatInvalid,

    #[error("Delete operation failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::delete_operation_failed))]
    DeleteOperationFailed { js_error: String },

    #[error("File deletion failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::file_deletion_failed))]
    FileDeletionFailed { js_error: String },

    #[error("Metadata delete operation failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::metadata_delete_failed))]
    MetadataDeleteFailed { js_error: String },

    #[error("Metadata deletion failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::metadata_deletion_failed))]
    MetadataDeletionFailed { js_error: String },

    #[error("GetAll request failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::get_all_request_failed))]
    GetAllRequestFailed { js_error: String },

    #[error("GetAll operation failed: {js_error}")]
    #[diagnostic(code(purr::web::storage::get_all_operation_failed))]
    GetAllOperationFailed { js_error: String },

    #[error("Array format invalid")]
    #[diagnostic(code(purr::web::storage::array_format_invalid))]
    ArrayFormatInvalid,
}

impl StorageError {
    /// Create a database open error with `JsValue`
    #[must_use]
    pub fn database_open_failed(js_error: JsValue) -> Self {
        Self::DatabaseOpenFailed {
            js_error: format_js_error(js_error),
        }
    }

    /// Create a database connection error with `JsValue`
    #[must_use]
    pub fn database_connection_failed(js_error: JsValue) -> Self {
        Self::DatabaseConnectionFailed {
            js_error: format_js_error(js_error),
        }
    }

    /// Create a transaction creation error with `JsValue`
    #[must_use]
    pub fn transaction_creation_failed(js_error: JsValue) -> Self {
        Self::TransactionCreationFailed {
            js_error: format_js_error(js_error),
        }
    }

    /// Create an object store access error with `JsValue`
    #[must_use]
    pub fn object_store_access_failed(js_error: JsValue) -> Self {
        Self::ObjectStoreAccessFailed {
            js_error: format_js_error(js_error),
        }
    }

    /// Create a file store error with `JsValue`
    #[must_use]
    pub fn file_store_failed(js_error: JsValue) -> Self {
        Self::FileStoreFailed {
            js_error: format_js_error(js_error),
        }
    }

    /// Create a metadata store error with `JsValue`
    #[must_use]
    pub fn metadata_store_failed(js_error: JsValue) -> Self {
        Self::MetadataStoreFailed {
            js_error: format_js_error(js_error),
        }
    }

    /// Create a get request error with `JsValue`
    #[must_use]
    pub fn get_request_failed(js_error: JsValue) -> Self {
        Self::GetRequestFailed {
            js_error: format_js_error(js_error),
        }
    }

    /// Create a get operation error with `JsValue`
    #[must_use]
    pub fn get_operation_failed(js_error: JsValue) -> Self {
        Self::GetOperationFailed {
            js_error: format_js_error(js_error),
        }
    }
}

/// Failures while turning an uploaded file into audio samples
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum AudioError {
    #[error("The audio file is empty")]
    #[diagnostic(code(purr::web::audio::empty))]
    Empty,

    #[error(
        "The audio file is too large ({} MB). The limit is {} MB.",
        .file_size.div_ceil(1_048_576),
        .max_size / 1_048_576
    )]
    #[diagnostic(code(purr::web::audio::too_large))]
    TooLarge { file_size: u64, max_size: u64 },

    #[error(
        "This browser could not decode the audio file. It may be corrupt or use a format the browser does not support; MP3 and WAV are decoded by every browser. ({message})"
    )]
    #[diagnostic(code(purr::web::audio::decode_failed))]
    DecodeFailed { message: String },

    #[error("The audio file decoded to no samples")]
    #[diagnostic(code(purr::web::audio::no_samples))]
    NoSamples,
}

/// A cached model file that does not match the manifest
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, miette::Diagnostic)]
pub enum IntegrityError {
    #[error("{path} is not in the browser cache")]
    #[diagnostic(code(purr::web::integrity::missing))]
    Missing { path: String },

    #[error("{path} has {actual} bytes in the cache but the manifest lists {expected}")]
    #[diagnostic(code(purr::web::integrity::size_mismatch))]
    SizeMismatch {
        path: String,
        expected: u64,
        actual: u64,
    },

    #[error("{path} failed its SHA-256 check (expected {expected}, got {actual})")]
    #[diagnostic(code(purr::web::integrity::digest_mismatch))]
    DigestMismatch {
        path: String,
        expected: String,
        actual: String,
    },
}

/// Web-specific errors that can occur during platform operations
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum WebError {
    #[error("IndexedDB operation failed: {operation}, {js_error}")]
    #[diagnostic(code(purr::web::indexeddb::operation_failed))]
    IndexedDb { operation: String, js_error: String },

    #[error("Web API error: {api} call failed, {js_error}")]
    #[diagnostic(code(purr::web::api::call_failed))]
    WebApi { api: String, js_error: String },

    #[error("JavaScript error: {context}, {js_error}")]
    #[diagnostic(code(purr::web::javascript::execution_error))]
    JavaScript { context: String, js_error: String },

    #[error("Model not found: {model_id} in {storage_type} storage")]
    #[diagnostic(code(purr::web::model::not_found))]
    ModelNotFound {
        model_id: String,
        storage_type: String,
    },

    #[error("Network error: {status_code} {status_text}")]
    #[diagnostic(code(purr::web::network::http_error))]
    NetworkError {
        status_code: u16,
        status_text: String,
        url: String,
    },

    #[error("Web Audio API error: {operation} failed, {js_error}")]
    #[diagnostic(code(purr::web::audio::api_failed))]
    WebAudioApi { operation: String, js_error: String },

    // === Transparent wrappers for specialized error enums ===
    #[error(transparent)]
    #[diagnostic(transparent)]
    Worker(WorkerError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Storage(StorageError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Audio(AudioError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Manifest(WebManifestError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Integrity(IntegrityError),
}

impl WebError {
    /// Create a new model not found error
    pub fn model_not_found(model_id: impl Into<String>, storage_type: impl Into<String>) -> Self {
        Self::ModelNotFound {
            model_id: model_id.into(),
            storage_type: storage_type.into(),
        }
    }

    /// Create a new network error
    pub fn network_error(
        status_code: u16,
        status_text: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        Self::NetworkError {
            status_code,
            status_text: status_text.into(),
            url: url.into(),
        }
    }

    /// Create a new `IndexedDB` error with `JsValue`
    pub fn indexeddb(operation: impl Into<String>, js_error: JsValue) -> Self {
        Self::IndexedDb {
            operation: operation.into(),
            js_error: format_js_error(js_error),
        }
    }

    /// Create a new Web API error with `JsValue`
    pub fn web_api(api: impl Into<String>, js_error: JsValue) -> Self {
        Self::WebApi {
            api: api.into(),
            js_error: format_js_error(js_error),
        }
    }

    /// Create a new Web Audio API error with `JsValue`
    pub fn web_audio_api(operation: impl Into<String>, js_error: JsValue) -> Self {
        Self::WebAudioApi {
            operation: operation.into(),
            js_error: format_js_error(js_error),
        }
    }

    /// Create a new JavaScript error with `JsValue`
    pub fn javascript(context: impl Into<String>, js_error: JsValue) -> Self {
        Self::JavaScript {
            context: context.into(),
            js_error: format_js_error(js_error),
        }
    }

    /// Create a `WebError` from a `JsValue`
    #[must_use]
    pub fn from_js_value(value: JsValue) -> Self {
        let message = if value.is_string() {
            value
                .as_string()
                .unwrap_or_else(|| "Unknown JS error".to_owned())
        } else {
            format!("{value:?}")
        };

        Self::JavaScript {
            context: "JavaScript execution".to_owned(),
            js_error: message,
        }
    }

    /// Convert to a `PlatformError`
    #[must_use]
    pub fn into_platform_error(self) -> PlatformError {
        match self {
            Self::Audio { .. } | Self::WebAudioApi { .. } => {
                PlatformError::audio_processing(Box::new(self))
            }
            Self::Worker { .. } => PlatformError::initialization(Box::new(self)),
            Self::Manifest { .. } | Self::Integrity { .. } => {
                PlatformError::model_management(Box::new(self))
            }
            Self::ModelNotFound { model_id, .. } => PlatformError::model_not_found(model_id),
            Self::Storage { .. }
            | Self::IndexedDb { .. }
            | Self::WebApi { .. }
            | Self::JavaScript { .. }
            | Self::NetworkError { .. } => PlatformError::io(Box::new(self)),
        }
    }
}

impl From<JsValue> for WebError {
    fn from(value: JsValue) -> Self {
        Self::from_js_value(value)
    }
}

impl From<WebError> for JsValue {
    fn from(error: WebError) -> Self {
        Self::from_str(&error.to_string())
    }
}

impl From<WebError> for PlatformError {
    fn from(error: WebError) -> Self {
        error.into_platform_error()
    }
}

// From implementations for specialized error enums
impl From<WorkerError> for WebError {
    fn from(error: WorkerError) -> Self {
        Self::Worker(error)
    }
}

impl From<StorageError> for WebError {
    fn from(error: StorageError) -> Self {
        Self::Storage(error)
    }
}

impl From<AudioError> for WebError {
    fn from(error: AudioError) -> Self {
        Self::Audio(error)
    }
}

impl From<IntegrityError> for WebError {
    fn from(error: IntegrityError) -> Self {
        Self::Integrity(error)
    }
}

impl From<WebManifestError> for WebError {
    fn from(error: WebManifestError) -> Self {
        Self::Manifest(error)
    }
}

impl From<UnknownWebModelError> for WebError {
    fn from(error: UnknownWebModelError) -> Self {
        Self::model_not_found(error.model_id, "manifest")
    }
}

/// Result type for web operations
pub type WebResult<T> = Result<T, WebError>;
