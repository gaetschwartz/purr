//! Error types for the web platform implementation

use purr_common::platform::PlatformError;
use wasm_bindgen::JsValue;

/// Helper function to format JsValue errors into strings while preserving information
pub(crate) fn format_js_error(js_error: JsValue) -> String {
    if js_error.is_string() {
        js_error
            .as_string()
            .unwrap_or_else(|| "Unknown JS error".to_string())
    } else if js_error.is_object() {
        // Try to get error message, name, and stack if available
        let obj = js_sys::Object::from(js_error);
        let mut parts = Vec::new();

        if let Ok(name) = js_sys::Reflect::get(&obj, &"name".into()) {
            if let Some(name_str) = name.as_string() {
                parts.push(format!("name: {}", name_str));
            }
        }

        if let Ok(message) = js_sys::Reflect::get(&obj, &"message".into()) {
            if let Some(message_str) = message.as_string() {
                parts.push(format!("message: {}", message_str));
            }
        }

        if let Ok(stack) = js_sys::Reflect::get(&obj, &"stack".into()) {
            if let Some(stack_str) = stack.as_string() {
                parts.push(format!("stack: {}", stack_str));
            }
        }

        if parts.is_empty() {
            format!("{:?}", obj)
        } else {
            parts.join(", ")
        }
    } else {
        format!("{:?}", js_error)
    }
}

// ========================================
// Specialized Error Enums for Domain-Specific Error Handling
// ========================================

/// Error for session not found scenarios
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
#[error("Session not found: {session_id}")]
#[diagnostic(code(purr::web::session::not_found))]
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

/// Worker-specific errors
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum WorkerError {
    #[error("Worker not initialized")]
    #[diagnostic(code(purr::web::worker::not_initialized))]
    NotInitialized,

    #[error("Worker command send failed")]
    #[diagnostic(code(purr::web::worker::command_send_failed))]
    CommandSendFailed,

    #[error("Worker ready timeout")]
    #[diagnostic(code(purr::web::worker::ready_timeout))]
    ReadyTimeout,

    #[error("Worker ready channel closed")]
    #[diagnostic(code(purr::web::worker::ready_channel_closed))]
    ReadyChannelClosed,

    #[error("Invalid request: {context}, {error_message}")]
    #[diagnostic(code(purr::web::worker::invalid_request))]
    InvalidRequest {
        context: String,
        error_message: String,
    },
}

/// Storage-specific errors for `IndexedDB` and other storage operations
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum StorageError {
    #[error("IndexedDB availability check failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::indexeddb_unavailable))]
    IndexedDbUnavailable { js_error: String },

    #[error("Database open failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::database_open_failed))]
    DatabaseOpenFailed { js_error: String },

    #[error("Database connection failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::database_connection_failed))]
    DatabaseConnectionFailed { js_error: String },

    #[error("Database cast failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::database_cast_failed))]
    DatabaseCastFailed { js_error: String },

    #[error("Transaction creation failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::transaction_creation_failed))]
    TransactionCreationFailed { js_error: String },

    #[error("Object store access failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::object_store_access_failed))]
    ObjectStoreAccessFailed { js_error: String },

    #[error("File store operation failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::file_store_failed))]
    FileStoreFailed { js_error: String },

    #[error("File storage operation failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::file_storage_operation_failed))]
    FileStorageOperationFailed { js_error: String },

    #[error("Metadata transaction creation failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::metadata_transaction_failed))]
    MetadataTransactionFailed { js_error: String },

    #[error("Metadata store access failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::metadata_store_access_failed))]
    MetadataStoreAccessFailed { js_error: String },

    #[error("Metadata serialization failed")]
    #[diagnostic(code(purr::web::storage::metadata_serialization_failed))]
    MetadataSerializationFailed,

    #[error("Metadata store operation failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::metadata_store_failed))]
    MetadataStoreFailed { js_error: String },

    #[error("Metadata storage operation failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::metadata_storage_operation_failed))]
    MetadataStorageOperationFailed { js_error: String },

    #[error("Get request failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::get_request_failed))]
    GetRequestFailed { js_error: String },

    #[error("Get operation failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::get_operation_failed))]
    GetOperationFailed { js_error: String },

    #[error("Data format invalid")]
    #[diagnostic(code(purr::web::storage::data_format_invalid))]
    DataFormatInvalid,

    #[error("Delete operation failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::delete_operation_failed))]
    DeleteOperationFailed { js_error: String },

    #[error("File deletion failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::file_deletion_failed))]
    FileDeletionFailed { js_error: String },

    #[error("Metadata delete operation failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::metadata_delete_failed))]
    MetadataDeleteFailed { js_error: String },

    #[error("Metadata deletion failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::metadata_deletion_failed))]
    MetadataDeletionFailed { js_error: String },

    #[error("GetAll request failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::get_all_request_failed))]
    GetAllRequestFailed { js_error: String },

    #[error("GetAll operation failed: {js_error:?}")]
    #[diagnostic(code(purr::web::storage::get_all_operation_failed))]
    GetAllOperationFailed { js_error: String },

    #[error("Array format invalid")]
    #[diagnostic(code(purr::web::storage::array_format_invalid))]
    ArrayFormatInvalid,
}

impl StorageError {
    /// Create a database open error with JsValue
    pub fn database_open_failed(js_error: JsValue) -> Self {
        Self::DatabaseOpenFailed {
            js_error: format_js_error(js_error),
        }
    }

    /// Create a database connection error with JsValue
    pub fn database_connection_failed(js_error: JsValue) -> Self {
        Self::DatabaseConnectionFailed {
            js_error: format_js_error(js_error),
        }
    }

    /// Create a transaction creation error with JsValue
    pub fn transaction_creation_failed(js_error: JsValue) -> Self {
        Self::TransactionCreationFailed {
            js_error: format_js_error(js_error),
        }
    }

    /// Create an object store access error with JsValue
    pub fn object_store_access_failed(js_error: JsValue) -> Self {
        Self::ObjectStoreAccessFailed {
            js_error: format_js_error(js_error),
        }
    }

    /// Create a file store error with JsValue
    pub fn file_store_failed(js_error: JsValue) -> Self {
        Self::FileStoreFailed {
            js_error: format_js_error(js_error),
        }
    }

    /// Create a metadata store error with JsValue
    pub fn metadata_store_failed(js_error: JsValue) -> Self {
        Self::MetadataStoreFailed {
            js_error: format_js_error(js_error),
        }
    }

    /// Create a get request error with JsValue
    pub fn get_request_failed(js_error: JsValue) -> Self {
        Self::GetRequestFailed {
            js_error: format_js_error(js_error),
        }
    }

    /// Create a get operation error with JsValue
    pub fn get_operation_failed(js_error: JsValue) -> Self {
        Self::GetOperationFailed {
            js_error: format_js_error(js_error),
        }
    }
}

/// Audio format and processing errors
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum AudioFormatError {
    #[error("File too small to analyze")]
    #[diagnostic(code(purr::web::audio::file_too_small_to_analyze))]
    FileTooSmallToAnalyze,

    #[error("File too small")]
    #[diagnostic(code(purr::web::audio::file_too_small))]
    FileTooSmall,

    #[error("Invalid WAV file")]
    #[diagnostic(code(purr::web::audio::invalid_wav_file))]
    InvalidWavFile,

    #[error("No MP3 data after ID3 tag")]
    #[diagnostic(code(purr::web::audio::no_mp3_data_after_id3))]
    NoMp3DataAfterId3,

    #[error("Invalid ID3/MP3 file")]
    #[diagnostic(code(purr::web::audio::invalid_id3_mp3_file))]
    InvalidId3Mp3File,

    #[error("Unknown or unsupported audio format")]
    #[diagnostic(code(purr::web::audio::unknown_audio_format))]
    UnknownAudioFormat,

    #[error("WAV file too small")]
    #[diagnostic(code(purr::web::audio::wav_file_too_small))]
    WavFileTooSmall,

    #[error("Invalid WAV header")]
    #[diagnostic(code(purr::web::audio::invalid_wav_header))]
    InvalidWavHeader,

    #[error("No fmt chunk found in WAV file")]
    #[diagnostic(code(purr::web::audio::no_fmt_chunk_found))]
    NoFmtChunkFound,

    #[error("Invalid fmt chunk size")]
    #[diagnostic(code(purr::web::audio::invalid_fmt_chunk_size))]
    InvalidFmtChunkSize,

    #[error("No MP3 frame found")]
    #[diagnostic(code(purr::web::audio::no_mp3_frame_found))]
    NoMp3FrameFound,

    #[error("No valid MP3 frame found")]
    #[diagnostic(code(purr::web::audio::no_valid_mp3_frame_found))]
    NoValidMp3FrameFound,

    #[error("Frame too small")]
    #[diagnostic(code(purr::web::audio::frame_too_small))]
    FrameTooSmall,

    #[error("Invalid sync word")]
    #[diagnostic(code(purr::web::audio::invalid_sync_word))]
    InvalidSyncWord,

    #[error("Invalid sample rate")]
    #[diagnostic(code(purr::web::audio::invalid_sample_rate))]
    InvalidSampleRate,

    #[error("Invalid channel mode")]
    #[diagnostic(code(purr::web::audio::invalid_channel_mode))]
    InvalidChannelMode,

    #[error("Invalid FLAC file")]
    #[diagnostic(code(purr::web::audio::invalid_flac_file))]
    InvalidFlacFile,

    #[error("FLAC file too small")]
    #[diagnostic(code(purr::web::audio::flac_file_too_small))]
    FlacFileTooSmall,

    #[error("First FLAC block is not STREAMINFO")]
    #[diagnostic(code(purr::web::audio::flac_first_block_not_streaminfo))]
    FlacFirstBlockNotStreaminfo,

    #[error("STREAMINFO block too small")]
    #[diagnostic(code(purr::web::audio::streaminfo_block_too_small))]
    StreaminfoBlockTooSmall,

    #[error("STREAMINFO data truncated")]
    #[diagnostic(code(purr::web::audio::streaminfo_data_truncated))]
    StreaminfoDataTruncated,

    #[error("No data chunk found in WAV file")]
    #[diagnostic(code(purr::web::audio::no_data_chunk_found))]
    NoDataChunkFound,

    #[error("Unsupported bit depth: {bit_depth}")]
    #[diagnostic(code(purr::web::audio::unsupported_bit_depth))]
    UnsupportedBitDepth { bit_depth: u16 },

    #[error("Empty audio file")]
    #[diagnostic(code(purr::web::audio::empty_audio_file))]
    EmptyAudioFile,

    #[error("File too small to be valid audio")]
    #[diagnostic(code(purr::web::audio::file_too_small_for_valid_audio))]
    FileTooSmallForValidAudio,

    #[error("Unsupported or invalid audio format")]
    #[diagnostic(code(purr::web::audio::unsupported_audio_format))]
    UnsupportedAudioFormat,
}

/// WebGPU-specific errors
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum WebGpuError {
    #[error("Missing limit: {limit_name}")]
    #[diagnostic(code(purr::web::webgpu::missing_limit))]
    MissingLimit { limit_name: String },

    #[error("Failed to get limit: {limit_name}")]
    #[diagnostic(code(purr::web::webgpu::failed_to_get_limit))]
    FailedToGetLimit { limit_name: String },

    #[error("Input and output buffer sizes don't match")]
    #[diagnostic(code(purr::web::webgpu::buffer_size_mismatch))]
    BufferSizeMismatch,

    #[error("Buffer data mismatch at index {index}: expected {expected}, got {actual}")]
    #[diagnostic(code(purr::web::webgpu::buffer_data_mismatch))]
    BufferDataMismatch {
        index: usize,
        expected: f32,
        actual: f32,
    },

    #[error("WebGPU computation failed: tolerance exceeded. Max difference: {max_diff}, tolerance: {tolerance}")]
    #[diagnostic(code(purr::web::webgpu::computation_failed))]
    ComputationFailed { max_diff: f32, tolerance: f32 },

    #[error("WebGPU feature not supported: {feature}")]
    #[diagnostic(code(purr::web::webgpu::feature_not_supported))]
    FeatureNotSupported { feature: String },
}

/// Web-specific errors that can occur during platform operations
#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum WebError {
    #[error("IndexedDB operation failed: {operation}, {js_error:?}")]
    #[diagnostic(code(purr::web::indexeddb::operation_failed))]
    IndexedDb { operation: String, js_error: String },

    #[error("Web API error: {api} call failed, {js_error:?}")]
    #[diagnostic(code(purr::web::api::call_failed))]
    WebApi { api: String, js_error: String },

    #[error("JavaScript error: {context}, {js_error:?}")]
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

    #[error("Audio format not supported: {format} (supported: {supported_formats})")]
    #[diagnostic(code(purr::web::audio::unsupported_format))]
    UnsupportedAudioFormat {
        format: String,
        supported_formats: String,
    },

    #[error("Audio file too large: {file_size} bytes exceeds limit of {max_size} bytes")]
    #[diagnostic(code(purr::web::audio::file_too_large))]
    AudioFileTooLarge { file_size: u64, max_size: u64 },

    #[error("Web Audio API error: {operation} failed, {js_error:?}")]
    #[diagnostic(code(purr::web::audio::api_failed))]
    WebAudioApi { operation: String, js_error: String },

    // === Transparent wrappers for specialized error enums ===
    #[error(transparent)]
    #[diagnostic(transparent)]
    SessionNotFound(SessionNotFoundError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Worker(WorkerError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Storage(StorageError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    AudioFormat(AudioFormatError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    WebGpu(WebGpuError),
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

    /// Create a new IndexedDB error with JsValue
    pub fn indexeddb(operation: impl Into<String>, js_error: JsValue) -> Self {
        Self::IndexedDb {
            operation: operation.into(),
            js_error: format_js_error(js_error),
        }
    }

    /// Create a new Web API error with JsValue
    pub fn web_api(api: impl Into<String>, js_error: JsValue) -> Self {
        Self::WebApi {
            api: api.into(),
            js_error: format_js_error(js_error),
        }
    }

    /// Create a new Web Audio API error with JsValue
    pub fn web_audio_api(operation: impl Into<String>, js_error: JsValue) -> Self {
        Self::WebAudioApi {
            operation: operation.into(),
            js_error: format_js_error(js_error),
        }
    }

    /// Create a new JavaScript error with JsValue
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
                .unwrap_or_else(|| "Unknown JS error".to_string())
        } else {
            format!("{value:?}")
        };

        WebError::JavaScript {
            context: "JavaScript execution".to_string(),
            js_error: message,
        }
    }

    /// Convert to a `PlatformError`
    #[must_use]
    pub fn into_platform_error(self) -> PlatformError {
        match self {
            WebError::AudioFormat { .. } => PlatformError::audio_processing(Box::new(self)),
            WebError::UnsupportedAudioFormat { .. } => {
                PlatformError::audio_processing(Box::new(self))
            }
            WebError::AudioFileTooLarge { .. } => PlatformError::audio_processing(Box::new(self)),
            WebError::WebAudioApi { .. } => PlatformError::audio_processing(Box::new(self)),
            WebError::WebGpu { .. } => PlatformError::initialization(Box::new(self)),
            WebError::Worker { .. } => PlatformError::initialization(Box::new(self)),
            WebError::Storage { .. } => PlatformError::io(Box::new(self)),
            WebError::IndexedDb { .. } => PlatformError::io(Box::new(self)),
            WebError::WebApi { .. } => PlatformError::io(Box::new(self)),
            WebError::JavaScript { .. } => PlatformError::io(Box::new(self)),
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

// From implementations for specialized error enums
impl From<WorkerError> for WebError {
    fn from(error: WorkerError) -> Self {
        WebError::Worker(error)
    }
}

impl From<StorageError> for WebError {
    fn from(error: StorageError) -> Self {
        WebError::Storage(error)
    }
}

impl From<AudioFormatError> for WebError {
    fn from(error: AudioFormatError) -> Self {
        WebError::AudioFormat(error)
    }
}

impl From<WebGpuError> for WebError {
    fn from(error: WebGpuError) -> Self {
        WebError::WebGpu(error)
    }
}

/// Result type for web operations
pub type WebResult<T> = Result<T, WebError>;
