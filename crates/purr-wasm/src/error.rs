//! Error types for the web platform implementation

use purr_common::platform::{IntegrityProblem, ModelIntegrity, PlatformError};
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
#[oopsie::oopsie]
pub enum WorkerError {
    #[oopsie(
        "The transcription worker was shut down before the request finished",
        code = "purr::web::worker::closed"
    )]
    Closed,

    #[oopsie("{message}", code = "purr::web::worker::request_failed")]
    RequestFailed {
        kind: crate::protocol::FailureKind,
        message: String,
    },
}

/// Storage-specific errors for `IndexedDB` and other storage operations
#[oopsie::oopsie]
pub enum StorageError {
    #[oopsie(
        "IndexedDB availability check failed: {js_error}",
        code = "purr::web::storage::indexeddb_unavailable"
    )]
    IndexedDbUnavailable { js_error: String },

    #[oopsie(
        "Database open failed: {js_error}",
        code = "purr::web::storage::database_open_failed"
    )]
    DatabaseOpenFailed { js_error: String },

    #[oopsie(
        "Database connection failed: {js_error}",
        code = "purr::web::storage::database_connection_failed"
    )]
    DatabaseConnectionFailed { js_error: String },

    #[oopsie(
        "Database cast failed: {js_error}",
        code = "purr::web::storage::database_cast_failed"
    )]
    DatabaseCastFailed { js_error: String },

    #[oopsie(
        "Transaction creation failed: {js_error}",
        code = "purr::web::storage::transaction_creation_failed"
    )]
    TransactionCreationFailed { js_error: String },

    #[oopsie(
        "Object store access failed: {js_error}",
        code = "purr::web::storage::object_store_access_failed"
    )]
    ObjectStoreAccessFailed { js_error: String },

    #[oopsie(
        "File store operation failed: {js_error}",
        code = "purr::web::storage::file_store_failed"
    )]
    FileStoreFailed { js_error: String },

    #[oopsie(
        "File storage operation failed: {js_error}",
        code = "purr::web::storage::file_storage_operation_failed"
    )]
    FileStorageOperationFailed { js_error: String },

    #[oopsie(
        "Metadata transaction creation failed: {js_error}",
        code = "purr::web::storage::metadata_transaction_failed"
    )]
    MetadataTransactionFailed { js_error: String },

    #[oopsie(
        "Metadata store access failed: {js_error}",
        code = "purr::web::storage::metadata_store_access_failed"
    )]
    MetadataStoreAccessFailed { js_error: String },

    #[oopsie(
        "Metadata serialization failed",
        code = "purr::web::storage::metadata_serialization_failed"
    )]
    MetadataSerializationFailed,

    #[oopsie(
        "Metadata store operation failed: {js_error}",
        code = "purr::web::storage::metadata_store_failed"
    )]
    MetadataStoreFailed { js_error: String },

    #[oopsie(
        "Metadata storage operation failed: {js_error}",
        code = "purr::web::storage::metadata_storage_operation_failed"
    )]
    MetadataStorageOperationFailed { js_error: String },

    #[oopsie(
        "Get request failed: {js_error}",
        code = "purr::web::storage::get_request_failed"
    )]
    GetRequestFailed { js_error: String },

    #[oopsie(
        "Get operation failed: {js_error}",
        code = "purr::web::storage::get_operation_failed"
    )]
    GetOperationFailed { js_error: String },

    #[oopsie(
        "Data format invalid",
        code = "purr::web::storage::data_format_invalid"
    )]
    DataFormatInvalid,

    #[oopsie(
        "Delete operation failed: {js_error}",
        code = "purr::web::storage::delete_operation_failed"
    )]
    DeleteOperationFailed { js_error: String },

    #[oopsie(
        "File deletion failed: {js_error}",
        code = "purr::web::storage::file_deletion_failed"
    )]
    FileDeletionFailed { js_error: String },

    #[oopsie(
        "Metadata delete operation failed: {js_error}",
        code = "purr::web::storage::metadata_delete_failed"
    )]
    MetadataDeleteFailed { js_error: String },

    #[oopsie(
        "Metadata deletion failed: {js_error}",
        code = "purr::web::storage::metadata_deletion_failed"
    )]
    MetadataDeletionFailed { js_error: String },

    #[oopsie(
        "GetAll request failed: {js_error}",
        code = "purr::web::storage::get_all_request_failed"
    )]
    GetAllRequestFailed { js_error: String },

    #[oopsie(
        "GetAll operation failed: {js_error}",
        code = "purr::web::storage::get_all_operation_failed"
    )]
    GetAllOperationFailed { js_error: String },

    #[oopsie(
        "Array format invalid",
        code = "purr::web::storage::array_format_invalid"
    )]
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
#[oopsie::oopsie]
pub enum AudioError {
    #[oopsie("The audio file is empty", code = "purr::web::audio::empty")]
    Empty,

    #[oopsie(
        "The audio file is too large ({} MB). The limit is {} MB.",
        file_size.div_ceil(1_048_576),
        max_size / 1_048_576,
        code = "purr::web::audio::too_large"
    )]
    TooLarge { file_size: u64, max_size: u64 },

    #[oopsie(
        "This browser could not decode the audio file. It may be corrupt or use a format the browser does not support; MP3 and WAV are decoded by every browser. ({message})",
        code = "purr::web::audio::decode_failed"
    )]
    DecodeFailed { message: String },

    #[oopsie(
        "The audio file decoded to no samples",
        code = "purr::web::audio::no_samples"
    )]
    NoSamples,
}

/// A cached model file that does not match the manifest
#[oopsie::oopsie]
#[derive(Clone, PartialEq, Eq)]
pub enum IntegrityError {
    #[oopsie(
        "{path} is not in the browser cache",
        code = "purr::web::integrity::missing"
    )]
    Missing { path: String },

    #[oopsie(
        "{path} has {actual} bytes in the cache but the manifest lists {expected}",
        code = "purr::web::integrity::size_mismatch"
    )]
    SizeMismatch {
        path: String,
        expected: u64,
        actual: u64,
    },

    #[oopsie(
        "{path} failed its SHA-256 check (expected {expected}, got {actual})",
        code = "purr::web::integrity::digest_mismatch"
    )]
    DigestMismatch {
        path: String,
        expected: String,
        actual: String,
    },
}

impl From<IntegrityError> for ModelIntegrity {
    fn from(error: IntegrityError) -> Self {
        let (problem, file) = match error {
            IntegrityError::Missing { path } => (IntegrityProblem::Missing, path),
            IntegrityError::SizeMismatch { path, .. } => (IntegrityProblem::SizeMismatch, path),
            IntegrityError::DigestMismatch { path, .. } => {
                (IntegrityProblem::ChecksumMismatch, path)
            }
        };
        Self::Damaged { problem, file }
    }
}

/// Web-specific errors that can occur during platform operations
#[oopsie::oopsie]
pub enum WebError {
    #[oopsie(
        "IndexedDB operation failed: {operation}, {js_error}",
        code = "purr::web::indexeddb::operation_failed"
    )]
    IndexedDb { operation: String, js_error: String },

    #[oopsie(
        "The browser has no storage left for {operation}",
        code = "purr::web::indexeddb::storage_full"
    )]
    StorageFull { operation: String },

    #[oopsie(
        "Web API error: {api} call failed, {js_error}",
        code = "purr::web::api::call_failed"
    )]
    WebApi { api: String, js_error: String },

    #[oopsie(
        "JavaScript error: {context}, {js_error}",
        code = "purr::web::javascript::execution_error"
    )]
    JavaScript { context: String, js_error: String },

    #[oopsie(
        "Model not found: {model_id} in {storage_type} storage",
        code = "purr::web::model::not_found"
    )]
    ModelNotFound {
        model_id: String,
        storage_type: String,
    },

    #[oopsie(
        "Network error: {status_code} {status_text}",
        code = "purr::web::network::http_error"
    )]
    NetworkError {
        status_code: u16,
        status_text: String,
        url: String,
    },

    #[oopsie(
        "Web Audio API error: {operation} failed, {js_error}",
        code = "purr::web::audio::api_failed"
    )]
    WebAudioApi { operation: String, js_error: String },

    #[oopsie(
        "Converting the model failed: {message}",
        code = "purr::web::model::conversion_failed"
    )]
    Conversion { message: String },

    #[oopsie("{source}")]
    Worker { source: WorkerError },

    #[oopsie(transparent)]
    Storage { source: StorageError },

    #[oopsie(transparent)]
    Audio { source: AudioError },

    #[oopsie(transparent)]
    Manifest { source: WebManifestError },

    #[oopsie(transparent)]
    Integrity { source: IntegrityError },
}

impl WebError {
    /// Create a new model not found error
    pub fn model_not_found(model_id: impl Into<String>, storage_type: impl Into<String>) -> Self {
        Self::ModelNotFound {
            model_id: model_id.into(),
            storage_type: storage_type.into(),
        }
    }

    pub fn conversion(message: impl Into<String>) -> Self {
        Self::Conversion {
            message: message.into(),
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
            Self::Manifest { .. } | Self::Integrity { .. } | Self::Conversion { .. } => {
                PlatformError::model_management(Box::new(self))
            }
            Self::ModelNotFound { model_id, .. } => PlatformError::model_not_found(model_id),
            Self::Storage { .. }
            | Self::IndexedDb { .. }
            | Self::StorageFull { .. }
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

impl From<UnknownWebModelError> for WebError {
    fn from(error: UnknownWebModelError) -> Self {
        Self::model_not_found(error.model_id, "manifest")
    }
}

/// Result type for web operations
pub type WebResult<T> = Result<T, WebError>;
