//! Error types for the web platform implementation

use purr_common::platform::PlatformError;
use wasm_bindgen::JsValue;

// ========================================
// Specialized Error Enums for Domain-Specific Error Handling
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

/// Worker-specific errors
#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("Worker not initialized")]
    NotInitialized,

    #[error("Worker command send failed: {details}")]
    CommandSendFailed { details: String },

    #[error("Worker ready timeout")]
    ReadyTimeout,

    #[error("Worker ready channel closed")]
    ReadyChannelClosed,

    #[error("Worker message handling failed: {details}")]
    MessageHandlingFailed { details: String },
}

/// Storage-specific errors for IndexedDB and other storage operations
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("IndexedDB availability check failed")]
    IndexedDbUnavailable,

    #[error("Database open failed")]
    DatabaseOpenFailed,

    #[error("Database connection failed")]
    DatabaseConnectionFailed,

    #[error("Database cast failed - invalid database object")]
    DatabaseCastFailed,

    #[error("Transaction creation failed")]
    TransactionCreationFailed,

    #[error("Object store access failed")]
    ObjectStoreAccessFailed,

    #[error("File store operation failed")]
    FileStoreFailed,

    #[error("File storage operation failed")]
    FileStorageOperationFailed,

    #[error("Metadata transaction creation failed")]
    MetadataTransactionFailed,

    #[error("Metadata store access failed")]
    MetadataStoreAccessFailed,

    #[error("Metadata serialization failed")]
    MetadataSerializationFailed,

    #[error("Metadata store operation failed")]
    MetadataStoreFailed,

    #[error("Metadata storage operation failed")]
    MetadataStorageOperationFailed,

    #[error("Get request failed")]
    GetRequestFailed,

    #[error("Get operation failed")]
    GetOperationFailed,

    #[error("Data format invalid")]
    DataFormatInvalid,

    #[error("Delete operation failed")]
    DeleteOperationFailed,

    #[error("File deletion failed")]
    FileDeletionFailed,

    #[error("Metadata delete operation failed")]
    MetadataDeleteFailed,

    #[error("Metadata deletion failed")]
    MetadataDeletionFailed,

    #[error("GetAll request failed")]
    GetAllRequestFailed,

    #[error("GetAll operation failed")]
    GetAllOperationFailed,

    #[error("Array format invalid")]
    ArrayFormatInvalid,
}

/// Audio format and processing errors
#[derive(Debug, thiserror::Error)]
pub enum AudioFormatError {
    #[error("File too small to analyze")]
    FileTooSmallToAnalyze,

    #[error("File too small")]
    FileTooSmall,

    #[error("Invalid WAV file")]
    InvalidWavFile,

    #[error("No MP3 data after ID3 tag")]
    NoMp3DataAfterId3,

    #[error("Invalid ID3/MP3 file")]
    InvalidId3Mp3File,

    #[error("Unknown or unsupported audio format")]
    UnknownAudioFormat,

    #[error("WAV file too small")]
    WavFileTooSmall,

    #[error("Invalid WAV header")]
    InvalidWavHeader,

    #[error("No fmt chunk found in WAV file")]
    NoFmtChunkFound,

    #[error("Invalid fmt chunk size")]
    InvalidFmtChunkSize,

    #[error("No MP3 frame found")]
    NoMp3FrameFound,

    #[error("No valid MP3 frame found")]
    NoValidMp3FrameFound,

    #[error("Frame too small")]
    FrameTooSmall,

    #[error("Invalid sync word")]
    InvalidSyncWord,

    #[error("Invalid sample rate")]
    InvalidSampleRate,

    #[error("Invalid channel mode")]
    InvalidChannelMode,

    #[error("Invalid FLAC file")]
    InvalidFlacFile,

    #[error("FLAC file too small")]
    FlacFileTooSmall,

    #[error("First FLAC block is not STREAMINFO")]
    FlacFirstBlockNotStreaminfo,

    #[error("STREAMINFO block too small")]
    StreaminfoBlockTooSmall,

    #[error("STREAMINFO data truncated")]
    StreaminfoDataTruncated,

    #[error("No data chunk found in WAV file")]
    NoDataChunkFound,

    #[error("Unsupported bit depth: {bit_depth}")]
    UnsupportedBitDepth { bit_depth: u16 },

    #[error("Resampling failed: source rate {source_rate}Hz, target rate {target_rate}Hz")]
    ResamplingFailed { source_rate: f32, target_rate: f32 },

    #[error("Empty audio file")]
    EmptyAudioFile,

    #[error("Sample rate {sample_rate}Hz is out of supported range (8000-48000 Hz)")]
    SampleRateOutOfRange { sample_rate: f32 },

    #[error("File too small to be valid audio")]
    FileTooSmallForValidAudio,

    #[error("Unsupported or invalid audio format")]
    UnsupportedAudioFormat,
}

/// WebGPU-specific errors
#[derive(Debug, thiserror::Error)]
pub enum WebGpuError {
    #[error("Missing limit: {limit_name}")]
    MissingLimit { limit_name: String },

    #[error("Failed to get limit: {limit_name}")]
    FailedToGetLimit { limit_name: String },

    #[error("Input and output buffer sizes don't match")]
    BufferSizeMismatch,

    #[error("Buffer data mismatch at index {index}: expected {expected}, got {actual}")]
    BufferDataMismatch { index: usize, expected: f32, actual: f32 },

    #[error("WebGPU computation failed: tolerance exceeded. Max difference: {max_diff}, tolerance: {tolerance}")]
    ComputationFailed { max_diff: f32, tolerance: f32 },

    #[error("WebGPU feature not supported: {feature}")]
    FeatureNotSupported { feature: String },
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

    #[error("Serialization error: {data_type} serialization failed")]
    Serialization {
        data_type: String,
        operation: String, // "serialize" or "deserialize"
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

    // === Transparent wrappers for specialized error enums ===
    #[error(transparent)]
    SessionNotFound(SessionNotFoundError),

    #[error(transparent)]
    Worker(WorkerError),

    #[error(transparent)]
    Storage(StorageError),

    #[error(transparent)]
    AudioFormat(AudioFormatError),

    #[error(transparent)]
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
    pub fn network_error(status_code: u16, status_text: impl Into<String>, url: impl Into<String>) -> Self {
        Self::NetworkError {
            status_code,
            status_text: status_text.into(),
            url: url.into(),
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
            WebError::AudioFormat { .. } => {
                PlatformError::audio_processing(Box::new(self))
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
            WebError::WebGpu { .. } => {
                PlatformError::initialization(Box::new(self))
            }
            WebError::Worker { .. } => {
                PlatformError::initialization(Box::new(self))
            }
            WebError::Storage { .. } => {
                PlatformError::io(Box::new(self))
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