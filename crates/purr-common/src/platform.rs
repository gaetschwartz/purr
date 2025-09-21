/// Platform abstraction layer for client-only architecture
/// This module provides a unified interface for platform-specific functionality
use bytes::Bytes;
use futures::Stream;
use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    collections::HashMap,
    ops::Deref,
    path::{Path, PathBuf},
    pin::Pin,
    str::FromStr,
};
// Note: tokio::sync primitives are used in platform implementations
#[allow(unused_imports)]
use tokio::sync::{Mutex, RwLock};

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

    #[error("Model management error: {source}")]
    #[diagnostic(code(platform::model_management))]
    ModelManagement {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Model not found: {model_id}")]
    #[diagnostic(code(platform::model_not_found))]
    ModelNotFound { model_id: String },

    #[error("Model download failed: {model_id} from {url}")]
    #[diagnostic(code(platform::model_download_failed))]
    ModelDownloadFailed {
        model_id: String,
        url: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Model installation failed: {model_id} at {path}")]
    #[diagnostic(code(platform::model_installation_failed))]
    ModelInstallationFailed {
        model_id: String,
        path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Invalid model metadata for {model_id}: {field} is {issue}")]
    #[diagnostic(code(platform::invalid_model_metadata))]
    InvalidModelMetadata {
        model_id: String,
        field: String,
        issue: String,
    },

    #[error("Model verification failed: {model_id} checksum mismatch")]
    #[diagnostic(code(platform::model_verification_failed))]
    ModelVerificationFailed {
        model_id: String,
        expected_checksum: String,
        actual_checksum: String,
    },

    #[error("Storage quota exceeded: {requested} bytes requested, {available} available")]
    #[diagnostic(code(platform::storage_quota_exceeded))]
    StorageQuotaExceeded {
        requested: u64,
        available: u64,
    },

    #[error(transparent)]
    #[diagnostic(transparent)]
    Other(#[from] UnsupportedPlatformError),
}

impl PlatformError {
    /// Create a new `PlatformError::FileProcessing`
    pub fn file_processing<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        PlatformError::FileProcessing { source: err.into() }
    }

    /// Create a new `PlatformError::Transcription`
    pub fn transcription<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        PlatformError::Transcription { source: err.into() }
    }

    /// Create a new `PlatformError::AudioProcessing`
    pub fn audio_processing<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        PlatformError::AudioProcessing { source: err.into() }
    }

    /// Create a new `PlatformError::Initialization`
    pub fn initialization<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        PlatformError::Initialization { source: err.into() }
    }

    /// Create a new `PlatformError::Io`
    pub fn io<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        PlatformError::Io { source: err.into() }
    }

    /// Create a new `PlatformError::Unsupported`
    pub fn unsupported<S>(operation: S) -> Self
    where
        S: Into<Cow<'static, str>>,
    {
        PlatformError::Unsupported {
            operation: operation.into(),
        }
    }

    /// Create a new `PlatformError::Unsupported`Platform
    pub fn unsupported_platform() -> Self {
        PlatformError::Other(UnsupportedPlatformError)
    }

    /// Create a new PlatformError::ModelManagement
    pub fn model_management<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        PlatformError::ModelManagement { source: err.into() }
    }

    /// Create a new PlatformError::ModelNotFound
    pub fn model_not_found<S>(model_id: S) -> Self
    where
        S: Into<String>,
    {
        PlatformError::ModelNotFound {
            model_id: model_id.into(),
        }
    }

    /// Create a new PlatformError::ModelDownloadFailed
    pub fn model_download_failed<S1, S2, E>(model_id: S1, url: S2, source: E) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        PlatformError::ModelDownloadFailed {
            model_id: model_id.into(),
            url: url.into(),
            source: source.into(),
        }
    }

    /// Create a new PlatformError::ModelInstallationFailed
    pub fn model_installation_failed<S, P, E>(model_id: S, path: P, source: E) -> Self
    where
        S: Into<String>,
        P: Into<PathBuf>,
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        PlatformError::ModelInstallationFailed {
            model_id: model_id.into(),
            path: path.into(),
            source: source.into(),
        }
    }

    /// Create a new PlatformError::InvalidModelMetadata
    pub fn invalid_model_metadata<S1, S2, S3>(model_id: S1, field: S2, issue: S3) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        PlatformError::InvalidModelMetadata {
            model_id: model_id.into(),
            field: field.into(),
            issue: issue.into(),
        }
    }

    /// Create a new PlatformError::ModelVerificationFailed
    pub fn model_verification_failed<S1, S2, S3>(
        model_id: S1,
        expected_checksum: S2,
        actual_checksum: S3,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        PlatformError::ModelVerificationFailed {
            model_id: model_id.into(),
            expected_checksum: expected_checksum.into(),
            actual_checksum: actual_checksum.into(),
        }
    }

    /// Create a new PlatformError::StorageQuotaExceeded
    pub fn storage_quota_exceeded(requested: u64, available: u64) -> Self {
        PlatformError::StorageQuotaExceeded {
            requested,
            available,
        }
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
    Error {
        operation: String,
        error_message: String,
    },
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
    Error {
        context: String,
        error_message: String,
    },
    /// Failed to init the transcription engine
    InitFailed {
        component: String,
        reason: String,
        error_details: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileId(String);

impl FileId {
    /// Create a new FileId
    pub fn new(id: String) -> Self {
        FileId(id)
    }
}

impl Deref for FileId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<Path> for FileId {
    fn as_ref(&self) -> &Path {
        Path::new(&self.0)
    }
}

impl std::fmt::Display for FileId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl<S> From<S> for FileId
where
    S: Into<String>,
{
    fn from(s: S) -> Self {
        FileId::new(s.into())
    }
}

impl FromStr for FileId {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(FileId::new(s.to_string()))
    }
}

/// Model information containing metadata about available models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    /// Unique identifier for the model
    pub id: String,
    /// Human-readable name of the model
    pub name: String,
    /// Description of the model's capabilities and characteristics
    pub description: String,
    /// Size of the model in bytes
    pub size_bytes: u64,
    /// Whether the model is currently installed/available locally
    pub is_installed: bool,
    /// Additional metadata specific to the model
    pub metadata: ModelMetadata,
}

impl ModelInfo {
    /// Create a new ModelInfo instance
    pub fn new(
        id: String,
        name: String,
        description: String,
        size_bytes: u64,
        is_installed: bool,
        metadata: ModelMetadata,
    ) -> Self {
        Self {
            id,
            name,
            description,
            size_bytes,
            is_installed,
            metadata,
        }
    }

    /// Get the model file path if installed
    pub fn local_path(&self) -> Option<&PathBuf> {
        if self.is_installed {
            self.metadata.local_path.as_ref()
        } else {
            None
        }
    }

    /// Mark model as installed with local path
    pub fn mark_installed(&mut self, local_path: PathBuf) {
        self.is_installed = true;
        self.metadata.local_path = Some(local_path);
    }

    /// Mark model as not installed
    pub fn mark_uninstalled(&mut self) {
        self.is_installed = false;
        self.metadata.local_path = None;
    }
}

/// Platform-specific metadata for models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMetadata {
    /// Download URL for the model (if available)
    pub download_url: Option<String>,
    /// Local file path where the model is stored (if installed)
    pub local_path: Option<PathBuf>,
    /// Model format (e.g., "ggml", "onnx", "webgpu")
    pub format: String,
    /// Model architecture type (e.g., "whisper", "transformer")
    pub architecture: String,
    /// Language support (e.g., "multilingual", "en-only")
    pub language_support: String,
    /// Quantization level (e.g., "q5_1", "q8_0", "fp16")
    pub quantization: Option<String>,
    /// Model version or revision
    pub version: String,
    /// Platform-specific attributes
    pub platform_specific: HashMap<String, String>,
}

impl ModelMetadata {
    /// Create new ModelMetadata instance
    pub fn new(
        format: String,
        architecture: String,
        language_support: String,
        version: String,
    ) -> Self {
        Self {
            download_url: None,
            local_path: None,
            format,
            architecture,
            language_support,
            quantization: None,
            version,
            platform_specific: HashMap::new(),
        }
    }

    /// Builder pattern for setting download URL
    pub fn with_download_url(mut self, url: String) -> Self {
        self.download_url = Some(url);
        self
    }

    /// Builder pattern for setting quantization
    pub fn with_quantization(mut self, quantization: String) -> Self {
        self.quantization = Some(quantization);
        self
    }

    /// Builder pattern for adding platform-specific attributes
    pub fn with_platform_attribute(mut self, key: String, value: String) -> Self {
        self.platform_specific.insert(key, value);
        self
    }
}

/// Progress information for model operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelOperationProgress {
    /// Operation starting
    Starting { model_id: String },
    /// Download in progress
    Downloading {
        model_id: String,
        bytes_downloaded: u64,
        total_bytes: Option<u64>,
        speed_bps: Option<u64>,
    },
    /// Installing/processing model
    Installing { model_id: String },
    /// Operation completed successfully
    Completed {
        model_id: String,
        local_path: PathBuf,
    },
    /// Operation failed
    Failed {
        model_id: String,
        operation: String,
        error_message: String,
    },
}

/// Platform trait defining the interface for platform-specific implementations
///
/// This trait provides a unified interface for both web and desktop platforms,
/// handling file processing, transcription, and model management operations.
///
/// # Platform Differences
///
/// ## Desktop Platform
/// - Uses local file system for model storage
/// - Can download models directly from internet
/// - Models stored in XDG-compliant directories
/// - Full featured model management with persistent storage
///
/// ## Web Platform
/// - Uses browser storage APIs (IndexedDB, OPFS) for model caching
/// - Models may be fetched on-demand or preloaded
/// - Limited by browser storage quotas
/// - May use WebGPU-optimized model formats
///
/// # Thread Safety
///
/// All implementations must be thread-safe and use tokio::sync primitives
/// for coordination between async operations. The trait requires Send + Sync.
#[async_trait::async_trait]
pub trait Platform: Send + Sync + 'static {
    async fn new() -> Result<Self, PlatformError>
    where
        Self: Sized;

    // ========================================
    // File Processing Operations
    // ========================================

    /// Process a file for transcription (handles temporary storage if needed)
    ///
    /// # Arguments
    /// * `file_data` - Raw file bytes
    /// * `file_path` - Original file path/name for context
    ///
    /// # Returns
    /// A unique FileId that can be used for subsequent operations
    async fn process_file(
        &self,
        file_data: Bytes,
        file_path: &Path,
    ) -> Result<FileId, PlatformError>;

    /// Start transcription of processed file
    ///
    /// # Arguments
    /// * `file_id` - ID returned from process_file
    /// * `request` - Transcription parameters
    ///
    /// # Returns
    /// A stream of transcription status updates
    async fn transcribe(
        &self,
        file_id: FileId,
        request: TranscriptionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<TranscriptionStatus, PlatformError>> + Send>>,
        PlatformError,
    >;

    /// Clean up temporary files if any
    ///
    /// # Arguments
    /// * `file_id` - ID of file to clean up
    async fn cleanup(&self, file_id: &str) -> Result<(), PlatformError>;

    // ========================================
    // Model Management Operations
    // ========================================

    /// List all models currently installed on the platform
    ///
    /// # Returns
    /// Vector of ModelInfo for installed models, thread-safe access
    ///
    /// # Platform Differences
    /// - **Desktop**: Scans local model directory, uses file system metadata
    /// - **Web**: Queries browser storage (IndexedDB/OPFS), checks cached models
    async fn list_installed_models(&self) -> Result<Vec<ModelInfo>, PlatformError>;

    /// List all models available for download/installation
    ///
    /// # Returns
    /// Vector of ModelInfo for available models, may include remote models
    ///
    /// # Platform Differences
    /// - **Desktop**: Returns full model catalog, can download any model
    /// - **Web**: May return filtered list based on browser capabilities,
    ///           WebGPU-optimized models preferred
    async fn list_available_models(&self) -> Result<Vec<ModelInfo>, PlatformError>;

    /// Fetch and install a model by ID with progress tracking
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier of the model to fetch
    ///
    /// # Returns
    /// Stream of progress updates during download and installation
    ///
    /// # Implementation Requirements
    /// - MUST use tokio::sync primitives for thread safety
    /// - MUST support concurrent downloads with proper synchronization
    /// - MUST handle partial downloads and resume capability
    /// - MUST validate model integrity after download
    ///
    /// # Platform Differences
    /// - **Desktop**: Downloads to local file system, uses reqwest for HTTP
    /// - **Web**: Uses fetch API, stores in browser storage, handles CORS
    async fn fetch_model(
        &self,
        model_id: &str,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ModelOperationProgress, PlatformError>> + Send>>,
        PlatformError,
    >;

    /// Get detailed information about a specific model
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier of the model
    ///
    /// # Returns
    /// ModelInfo with current installation status and metadata
    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo, PlatformError>;

    /// Remove an installed model from the platform
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier of the model to remove
    ///
    /// # Platform Differences
    /// - **Desktop**: Deletes model file from file system
    /// - **Web**: Removes from browser storage, clears cache entries
    async fn remove_model(&self, model_id: &str) -> Result<(), PlatformError>;

    /// Check if a specific model is currently installed
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier of the model to check
    ///
    /// # Returns
    /// True if model is installed and ready for use
    async fn is_model_installed(&self, model_id: &str) -> Result<bool, PlatformError>;

    /// Get the local path or identifier for an installed model
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier of the model
    ///
    /// # Returns
    /// Platform-specific path or identifier for accessing the model
    ///
    /// # Platform Differences
    /// - **Desktop**: Returns file system path to model file
    /// - **Web**: Returns storage key or blob URL for browser access
    async fn get_model_path(&self, model_id: &str) -> Result<String, PlatformError>;
}
