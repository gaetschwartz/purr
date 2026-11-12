use ambassador::delegatable_trait;
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

use crate::{
    accel::{AccelPolicy, LastRun, SystemReport},
    audio_overview::AudioOverview,
    browser_engine::BrowserEngine,
    history::{
        entry::{HistoryEntry, HistoryId, HistoryRecord},
        listing::{HistoryListing, Saved},
    },
    hugging_face_token::HuggingFaceToken,
    settings::Settings,
};

pub type TranscriptionStream =
    Pin<Box<dyn Stream<Item = Result<TranscriptionStatus, PlatformError>> + Send>>;

pub type ModelProgressStream =
    Pin<Box<dyn Stream<Item = Result<ModelOperationProgress, PlatformError>> + Send>>;

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
    StorageQuotaExceeded { requested: u64, available: u64 },

    #[error("Device detection error: {source}")]
    #[diagnostic(code(platform::device_detection))]
    DeviceDetection {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("{source}")]
    #[diagnostic(code(platform::settings_persistence))]
    SettingsPersistence {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("{source}")]
    #[diagnostic(code(platform::history))]
    History {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("There is no saved transcription {id}")]
    #[diagnostic(code(platform::history_entry_not_found))]
    HistoryEntryNotFound { id: HistoryId },

    #[error(transparent)]
    #[diagnostic(transparent)]
    UnsupportedPlatform(#[from] UnsupportedPlatformError),
}

impl PlatformError {
    /// Create a new `PlatformError::FileProcessing`
    pub fn file_processing<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::FileProcessing { source: err.into() }
    }

    /// Create a new `PlatformError::Transcription`
    pub fn transcription<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::Transcription { source: err.into() }
    }

    /// Create a new `PlatformError::AudioProcessing`
    pub fn audio_processing<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::AudioProcessing { source: err.into() }
    }

    /// Create a new `PlatformError::Initialization`
    pub fn initialization<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::Initialization { source: err.into() }
    }

    /// Create a new `PlatformError::Io`
    pub fn io<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::Io { source: err.into() }
    }

    /// Create a new `PlatformError::Unsupported`
    pub fn unsupported<S>(operation: S) -> Self
    where
        S: Into<Cow<'static, str>>,
    {
        Self::Unsupported {
            operation: operation.into(),
        }
    }

    /// Create a new `PlatformError::Unsupported`Platform
    #[must_use]
    pub fn unsupported_platform() -> Self {
        Self::UnsupportedPlatform(UnsupportedPlatformError)
    }

    /// Create a new `PlatformError::ModelManagement`
    pub fn model_management<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::ModelManagement { source: err.into() }
    }

    /// Create a new `PlatformError::ModelNotFound`
    pub fn model_not_found<S>(model_id: S) -> Self
    where
        S: Into<String>,
    {
        Self::ModelNotFound {
            model_id: model_id.into(),
        }
    }

    /// Create a new `PlatformError::ModelDownloadFailed`
    pub fn model_download_failed<S1, S2, E>(model_id: S1, url: S2, source: E) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::ModelDownloadFailed {
            model_id: model_id.into(),
            url: url.into(),
            source: source.into(),
        }
    }

    /// Create a new `PlatformError::ModelInstallationFailed`
    pub fn model_installation_failed<S, P, E>(model_id: S, path: P, source: E) -> Self
    where
        S: Into<String>,
        P: Into<PathBuf>,
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::ModelInstallationFailed {
            model_id: model_id.into(),
            path: path.into(),
            source: source.into(),
        }
    }

    /// Create a new `PlatformError::InvalidModelMetadata`
    pub fn invalid_model_metadata<S1, S2, S3>(model_id: S1, field: S2, issue: S3) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        S3: Into<String>,
    {
        Self::InvalidModelMetadata {
            model_id: model_id.into(),
            field: field.into(),
            issue: issue.into(),
        }
    }

    /// Create a new `PlatformError::ModelVerificationFailed`
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
        Self::ModelVerificationFailed {
            model_id: model_id.into(),
            expected_checksum: expected_checksum.into(),
            actual_checksum: actual_checksum.into(),
        }
    }

    /// Create a new `PlatformError::StorageQuotaExceeded`
    #[must_use]
    pub fn storage_quota_exceeded(requested: u64, available: u64) -> Self {
        Self::StorageQuotaExceeded {
            requested,
            available,
        }
    }

    /// Create a new `PlatformError::DeviceDetection`
    pub fn device_detection<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::DeviceDetection { source: err.into() }
    }

    /// Create a new `PlatformError::SettingsPersistence`
    pub fn settings_persistence<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::SettingsPersistence { source: err.into() }
    }

    /// Create a new `PlatformError::History`
    pub fn history<E>(err: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::History { source: err.into() }
    }
}

#[derive(Debug, Clone, Copy, thiserror::Error, Diagnostic, PartialEq, Eq)]
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

/// Transcription request parameters
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    /// File to transcribe (bytes, path, or uploaded file ID)
    pub file: FileSource,
    /// The model to transcribe with, as listed by [`Platform::list_available_models`]
    pub model: ModelId,
    /// Optional language (auto-detect if None)
    pub language: Option<String>,
    /// Whether to translate to English
    pub translate: bool,
    /// How the engine runs; platforms that do not choose their own hardware ignore it
    pub engine: EngineOptions,
    /// Authenticates the downloads of a model that is not stored yet; desktop never downloads here
    pub hugging_face_token: Option<HuggingFaceToken>,
}

/// How the transcription engine runs, taken from the performance and model settings.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EngineOptions {
    /// Which hardware to run on
    pub accel: AccelPolicy,
    /// CPU threads, `None` to let the engine choose
    pub threads: Option<u32>,
    /// Sampling temperature, 0.0 being deterministic
    pub temperature: f64,
}

impl EngineOptions {
    /// The options the user's settings ask for.
    #[must_use]
    pub const fn from_settings(settings: &Settings) -> Self {
        Self {
            accel: settings.performance.accel_policy(),
            threads: settings.performance.threads,
            temperature: settings.model.temperature,
        }
    }
}

#[derive(Debug, Clone)]
pub enum FileSource {
    Bytes(Bytes),
    Path(PathBuf),
    Uploaded(FileId),
}

impl std::fmt::Display for FileSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Path(p) => write!(f, "[{}]", p.display()),
            Self::Bytes(b) => write!(f, "[{} bytes]", b.len()),
            Self::Uploaded(id) => write!(f, "[#{id}]"),
        }
    }
}

impl From<PathBuf> for FileSource {
    fn from(path: PathBuf) -> Self {
        Self::Path(path)
    }
}

impl From<&Path> for FileSource {
    fn from(path: &Path) -> Self {
        Self::Path(path.to_path_buf())
    }
}

impl From<FileId> for FileSource {
    fn from(file_id: FileId) -> Self {
        Self::Uploaded(file_id)
    }
}

impl From<Bytes> for FileSource {
    fn from(bytes: Bytes) -> Self {
        Self::Bytes(bytes)
    }
}

/// A timed piece of a transcript
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TranscriptSegment {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
}

/// Transcription status updates
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TranscriptionStatus {
    /// Starting transcription
    Starting,
    /// Processing audio file
    ProcessingAudio,
    /// Model files are being downloaded or loaded into memory
    LoadingModel { loaded_bytes: u64, total_bytes: u64 },
    /// The model is ready and speech is being decoded; segments follow
    Transcribing,
    /// The hardware the engine settled on, reported once the model has loaded
    Engine { run: LastRun },
    /// The execution provider a browser run settled on, reported before the model runs
    BrowserEngine { engine: BrowserEngine },
    /// The platform recognised the spoken language (a Whisper language code)
    LanguageDetected { language: String },
    /// Transcription in progress; `text` is the whole current text of segment `chunk_index`
    InProgress {
        chunk_index: usize,
        text: String,
        start_time: f64,
        end_time: f64,
    },
    /// Authoritative segment list that replaces everything reported through `InProgress`
    Segments { segments: Vec<TranscriptSegment> },
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
    /// Create a new `FileId`
    #[must_use]
    pub fn new(id: String) -> Self {
        Self(id)
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
        Self::new(s.into())
    }
}

impl FromStr for FileId {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::new(s.to_owned()))
    }
}

/// The id of a model in the platform's catalog, as [`ModelInfo::id`] carries it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ModelId(String);

impl ModelId {
    /// Wraps a catalog id.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The id as the catalog spells it.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Model information containing metadata about available models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    /// Create a new `ModelInfo` instance
    #[must_use]
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
    #[must_use]
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
}

/// Keys of the flags a platform can set in [`ModelMetadata::platform_specific`].
pub mod model_flag {
    /// The model needs the WebGPU `shader-f16` feature.
    pub const REQUIRES_SHADER_F16: &str = "requires_shader_f16";
    /// The model only transcribes English.
    pub const ENGLISH_ONLY: &str = "english_only";
    /// The model marks where the speaker changes.
    pub const SPEAKER_TURNS: &str = "speaker_turns";
}

/// Platform-specific metadata for models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    /// Quantization level (e.g., "`q5_1`", "`q8_0`", "fp16")
    pub quantization: Option<String>,
    /// Model version or revision
    pub version: String,
    /// Platform-specific attributes
    pub platform_specific: HashMap<String, String>,
}

impl ModelMetadata {
    /// Create new `ModelMetadata` instance
    #[must_use]
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
    #[must_use]
    pub fn with_download_url(mut self, url: String) -> Self {
        self.download_url = Some(url);
        self
    }

    /// Builder pattern for setting quantization
    #[must_use]
    pub fn with_quantization(mut self, quantization: String) -> Self {
        self.quantization = Some(quantization);
        self
    }

    /// Whether the platform marked the model with `flag` (see [`model_flag`]).
    #[must_use]
    pub fn has_flag(&self, flag: &str) -> bool {
        self.platform_specific
            .get(flag)
            .is_some_and(|value| value == "true")
    }

    /// Builder pattern for adding platform-specific attributes
    #[must_use]
    pub fn with_platform_attribute(mut self, key: String, value: String) -> Self {
        self.platform_specific.insert(key, value);
        self
    }
}

/// Progress information for model operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
        #[serde(default)]
        kind: ModelFailureKind,
    },
}

/// What went wrong in a failed model operation, so the UI can word the recovery.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ModelFailureKind {
    /// The connection to the model host failed or dropped.
    Network,
    /// The downloaded or cached files do not match the published checksums.
    Integrity,
    /// Anything else, including storage errors.
    #[default]
    Other,
}

/// How a stored model file differs from its published description.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IntegrityProblem {
    /// A file of the model is not stored.
    Missing,
    /// A file has a different size than published.
    SizeMismatch,
    /// A file's SHA-256 differs from the published checksum.
    ChecksumMismatch,
}

/// The outcome of checking an installed model against its published checksums.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelIntegrity {
    /// Every file matches.
    Verified,
    /// A file does not match; `file` names the first one that failed.
    Damaged {
        problem: IntegrityProblem,
        file: String,
    },
}

/// Device type for transcription acceleration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeviceType {
    /// CPU device - always available
    Cpu,
    /// GPU device - available when GPU acceleration is supported
    Gpu,
    /// Accelerator device (TPU, NPU, etc.)
    Accel,
    /// Unknown device type
    Unknown,
}

/// Information about available devices for transcription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Unique device identifier
    pub id: i32,
    /// Device name; empty when the platform gives none
    pub name: String,
    /// Device description (optional)
    pub description: Option<String>,
    /// Device type
    pub device_type: DeviceType,
    /// Free memory in bytes (None if unknown)
    pub memory_free: Option<usize>,
    /// Total memory in bytes (None if unknown)
    pub memory_total: Option<usize>,
    /// Device capabilities (platform-specific)
    pub capabilities: Option<HashMap<String, serde_json::Value>>,
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
/// - Uses browser storage APIs (`IndexedDB`, OPFS) for model caching
/// - Models may be fetched on-demand or preloaded
/// - Limited by browser storage quotas
/// - May use WebGPU-optimized model formats
///
/// # Thread Safety
///
/// All implementations must be thread-safe and use `tokio::sync` primitives
/// for coordination between async operations. The trait requires Send + Sync.
#[delegatable_trait]
pub trait Platform: Send + Sync + 'static {
    fn install_logging_hooks(&self);
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
    /// A unique `FileId` that can be used for subsequent operations
    fn process_file(
        &self,
        file_data: Bytes,
        file_path: &Path,
    ) -> impl std::future::Future<Output = Result<FileId, PlatformError>> + Send;

    /// Start transcription of processed file
    ///
    /// # Arguments
    /// * `file_id` - ID returned from `process_file`
    /// * `request` - Transcription parameters
    ///
    /// # Returns
    /// A stream of transcription status updates
    fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> impl std::future::Future<Output = Result<TranscriptionStream, PlatformError>> + Send;

    /// Clean up temporary files if any
    ///
    /// # Arguments
    /// * `file_id` - ID of file to clean up
    fn cleanup(
        &self,
        file_id: &str,
    ) -> impl std::future::Future<Output = Result<(), PlatformError>> + Send;

    /// The largest upload the platform accepts, in bytes, when it has a limit
    fn max_upload_bytes(&self) -> Option<u64>;

    /// Measure the length and waveform of a processed file
    ///
    /// # Arguments
    /// * `file_id` - ID returned from `process_file`
    /// * `buckets` - How many waveform bars to draw
    ///
    /// # Platform Differences
    /// - **Desktop**: Reads the stored file through the chunk stream the transcription reads
    /// - **Web**: Decodes the stored file with the Web Audio API, as the transcription does
    ///
    /// # Implementation Requirements
    /// - MUST NOT keep the decoded samples once the overview is made
    fn audio_overview(
        &self,
        file_id: &FileId,
        buckets: usize,
    ) -> impl std::future::Future<Output = Result<AudioOverview, PlatformError>> + Send;

    // ========================================
    // History Operations
    // ========================================

    /// List the saved transcriptions, newest first
    ///
    /// # Returns
    /// The readable entries as summaries and the stored entries that could not be read; one
    /// unreadable entry never fails the listing.
    ///
    /// # Platform Differences
    /// - **Desktop**: One JSON file per entry in the app's data directory
    /// - **Web**: One JSON string per entry in an `IndexedDB` object store
    fn list_history(
        &self,
    ) -> impl std::future::Future<Output = Result<HistoryListing, PlatformError>> + Send;

    /// Load a saved transcription with whether its audio can still be played
    ///
    /// # Returns
    /// `PlatformError::HistoryEntryNotFound` when there is no such entry
    fn get_history_entry(
        &self,
        id: &HistoryId,
    ) -> impl std::future::Future<Output = Result<HistoryRecord, PlatformError>> + Send;

    /// Store an entry, replacing one with the same id, then enforce the retention limits
    ///
    /// # Returns
    /// The entries dropped to stay within the limits, which can include `entry` itself when it
    /// is older than everything that is kept
    fn save_history_entry(
        &self,
        entry: &HistoryEntry,
    ) -> impl std::future::Future<Output = Result<Saved, PlatformError>> + Send;

    /// Change the title of a saved transcription; `title` must already be a valid one
    fn rename_history_entry(
        &self,
        id: &HistoryId,
        title: &str,
    ) -> impl std::future::Future<Output = Result<(), PlatformError>> + Send;

    /// Delete a saved transcription; the audio file it points at is not touched
    fn delete_history_entry(
        &self,
        id: &HistoryId,
    ) -> impl std::future::Future<Output = Result<(), PlatformError>> + Send;

    /// Delete every saved transcription, including unreadable ones
    ///
    /// # Returns
    /// How many stored entries were removed
    fn clear_history(
        &self,
    ) -> impl std::future::Future<Output = Result<usize, PlatformError>> + Send;

    // ========================================
    // Model Management Operations
    // ========================================

    /// List all models currently installed on the platform
    ///
    /// # Returns
    /// Vector of `ModelInfo` for installed models, thread-safe access
    ///
    /// # Platform Differences
    /// - **Desktop**: Scans local model directory, uses file system metadata
    /// - **Web**: Queries browser storage (IndexedDB/OPFS), checks cached models
    fn list_installed_models(
        &self,
    ) -> impl std::future::Future<Output = Result<Vec<ModelInfo>, PlatformError>> + Send;

    /// List all models available for download/installation
    ///
    /// # Returns
    /// Vector of `ModelInfo` for available models, may include remote models
    ///
    /// # Platform Differences
    /// - **Desktop**: Returns full model catalog, can download any model
    /// - **Web**: May return filtered list based on browser capabilities,
    ///           WebGPU-optimized models preferred
    fn list_available_models(
        &self,
    ) -> impl std::future::Future<Output = Result<Vec<ModelInfo>, PlatformError>> + Send;

    /// Fetch and install a model by ID with progress tracking
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier of the model to fetch
    /// * `token` - Authenticates the requests to Hugging Face, when the user set one
    ///
    /// # Returns
    /// Stream of progress updates during download and installation
    ///
    /// # Implementation Requirements
    /// - MUST use `tokio::sync` primitives for thread safety
    /// - MUST support concurrent downloads with proper synchronization
    /// - MUST handle partial downloads and resume capability
    /// - MUST validate model integrity after download
    ///
    /// # Platform Differences
    /// - **Desktop**: Downloads to local file system, uses reqwest for HTTP
    /// - **Web**: Uses fetch API, stores in browser storage, handles CORS
    fn fetch_model(
        &self,
        model_id: &str,
        token: Option<HuggingFaceToken>,
    ) -> impl std::future::Future<Output = Result<ModelProgressStream, PlatformError>> + Send;

    /// Get detailed information about a specific model
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier of the model
    ///
    /// # Returns
    /// `ModelInfo` with current installation status and metadata
    fn get_model_info(
        &self,
        model_id: &str,
    ) -> impl std::future::Future<Output = Result<ModelInfo, PlatformError>> + Send;

    /// Remove an installed model from the platform
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier of the model to remove
    ///
    /// # Platform Differences
    /// - **Desktop**: Deletes model file from file system
    /// - **Web**: Removes from browser storage, clears cache entries
    fn remove_model(
        &self,
        model_id: &str,
    ) -> impl std::future::Future<Output = Result<(), PlatformError>> + Send;

    /// Hash an installed model's files against the published checksums
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier of the model to check
    ///
    /// # Returns
    /// `ModelIntegrity::Damaged` when a file does not match; `Err` only when the check itself
    /// could not run.
    ///
    /// # Platform Differences
    /// - **Desktop**: Hashes the model file on disk
    /// - **Web**: Hashes every cached file and evicts the ones that fail, so the next use
    ///   downloads them again
    fn verify_model(
        &self,
        model_id: &str,
    ) -> impl std::future::Future<Output = Result<ModelIntegrity, PlatformError>> + Send;

    /// Check if a specific model is currently installed
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier of the model to check
    ///
    /// # Returns
    /// True if model is installed and ready for use
    fn is_model_installed(
        &self,
        model_id: &str,
    ) -> impl std::future::Future<Output = Result<bool, PlatformError>> + Send;

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
    fn get_model_path(
        &self,
        model_id: &str,
    ) -> impl std::future::Future<Output = Result<String, PlatformError>> + Send;

    /// List available devices for transcription
    ///
    /// # Returns
    /// Vector of available devices with their capabilities
    ///
    /// # Platform Differences
    /// - **Desktop**: Uses GPU detection from purr-core, includes CPU and available GPUs
    /// - **Web**: Always includes CPU, includes GPU only when WebGPU is available
    ///
    /// # Implementation Notes
    /// - Memory and capability values should be set to None if not accurately determinable
    /// - Prefer conservative estimates over potentially incorrect values
    fn list_available_devices(
        &self,
    ) -> impl std::future::Future<Output = Result<Vec<DeviceInfo>, PlatformError>> + Send;

    /// Report the acceleration backends this machine offers, best first
    ///
    /// # Platform Differences
    /// - **Desktop**: Probes the backends built into purr-core and lists their devices
    /// - **Web**: Unsupported; the browser's own GPU is reported by [`Self::list_available_devices`]
    fn system_report(
        &self,
    ) -> impl std::future::Future<Output = Result<SystemReport, PlatformError>> + Send;

    // ========================================
    // Settings Persistence Operations
    // ========================================

    /// Load settings from persistent storage
    ///
    /// # Returns
    /// Settings loaded from platform-specific storage, or default settings if none exist
    ///
    /// # Platform Differences
    /// - **Desktop**: Loads from XDG-compliant config directory (e.g., `~/.config/purr/settings.json`)
    /// - **Web**: Loads from browser localStorage or IndexedDB
    ///
    /// # Error Handling
    /// - Returns default settings if no saved settings exist
    /// - Returns `PlatformError::SettingsPersistence` for I/O or parsing errors
    /// - Performs automatic migration for older settings versions
    fn load_settings(
        &self,
    ) -> impl std::future::Future<Output = Result<Settings, PlatformError>> + Send;

    /// Save settings to persistent storage
    ///
    /// # Arguments
    /// * `settings` - Settings configuration to persist
    ///
    /// # Platform Differences
    /// - **Desktop**: Saves to XDG-compliant config directory with atomic writes
    /// - **Web**: Saves to browser localStorage or IndexedDB with transaction safety
    ///
    /// # Implementation Requirements
    /// - MUST use atomic writes/transactions to prevent corruption
    /// - MUST validate settings before saving
    /// - MUST create parent directories if they don't exist (desktop)
    /// - SHOULD backup existing settings before overwriting
    fn save_settings(
        &self,
        settings: &Settings,
    ) -> impl std::future::Future<Output = Result<(), PlatformError>> + Send;

    /// Get the storage location for settings (for debugging and user information)
    ///
    /// # Returns
    /// Platform-specific path or identifier where settings are stored
    ///
    /// # Platform Differences
    /// - **Desktop**: Returns file system path (e.g., `/home/user/.config/purr/settings.json`)
    /// - **Web**: Returns storage key or description (e.g., `localStorage:purr-settings`)
    ///
    /// # Implementation Notes
    /// - This method is primarily for debugging and user information
    /// - Should return a human-readable representation of the storage location
    /// - May return `None` if the location is not deterministic or accessible
    fn get_settings_location(
        &self,
    ) -> impl std::future::Future<Output = Result<Option<String>, PlatformError>> + Send;
}

#[cfg(test)]
mod engine_options_tests {
    use super::*;
    use crate::accel::{AccelBackend, AccelSelection};

    #[test]
    fn the_engine_follows_the_performance_and_model_settings() {
        let mut settings = Settings::default();
        settings.performance.accel_backend = AccelSelection::Vulkan;
        settings.performance.accel_fallback = false;
        settings.performance.threads = Some(3);
        settings.model.temperature = 0.4;
        assert_eq!(
            EngineOptions::from_settings(&settings),
            EngineOptions {
                accel: AccelPolicy::Forced {
                    backend: AccelBackend::Vulkan,
                    allow_fallback: false
                },
                threads: Some(3),
                temperature: 0.4,
            }
        );
    }

    #[test]
    fn the_default_settings_leave_the_engine_to_choose() {
        assert_eq!(
            EngineOptions::from_settings(&Settings::default()),
            EngineOptions::default()
        );
    }
}
