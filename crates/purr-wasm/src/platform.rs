//! Web platform implementation of the Platform trait

use crate::error::{WebError, WebResult};
use crate::model::WebModelManager;
use crate::storage::WebStorage;
use crate::transcription::{
    start_transcription_process, validate_audio_file, AudioProcessingConfig,
};
use crate::worker::{TranscriptionConfig, TranscriptionWorker};
use bytes::Bytes;
use futures::Stream;
use futures::StreamExt;
use purr_common::platform::{
    DeviceInfo, DeviceType, FileId, FileSource, ModelInfo, ModelMetadata, ModelOperationProgress,
    Platform, PlatformError, TranscriptionRequest, TranscriptionStatus,
};
use std::collections::HashMap;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{oneshot, Mutex, RwLock};
use wasm_bindgen_futures;
use web_sys::GpuAdapter;

/// Web platform implementation using WebAssembly and browser APIs
pub struct PlatformImpl {
    /// Storage manager for file operations
    storage: Arc<WebStorage>,
    /// Model manager for model operations
    model_manager: Arc<WebModelManager>,
    /// Worker pool for transcription
    worker: Arc<TranscriptionWorker>,
    /// Active transcription sessions
    sessions: RwLock<HashMap<String, String>>,
    /// Platform initialization status
    initialized: RwLock<bool>,
    /// Configuration mutex for thread safety
    config_mutex: Mutex<()>,
}

impl PlatformImpl {
    /// Create a new `WebPlatform` instance
    pub fn new() -> WebResult<Self> {
        let storage = Arc::new(WebStorage::new());
        let model_manager = Arc::new(WebModelManager::with_storage(storage.clone()));
        let worker = Arc::new(TranscriptionWorker::new(model_manager.clone()));

        let platform = Self {
            storage,
            model_manager,
            worker,
            sessions: RwLock::new(HashMap::new()),
            initialized: RwLock::new(false),
            config_mutex: Mutex::new(()),
        };

        // Initialize default models
        // Defer initialization to first use to avoid async in constructor
        // The platform will initialize when first needed

        Ok(platform)
    }

    /// Initialize default models available for web platform
    async fn initialize_default_models(&self) -> WebResult<()> {
        let default_models = vec![
            ModelInfo::new(
                "whisper-tiny".to_string(),
                "Whisper Tiny".to_string(),
                "Smallest Whisper model, fastest but least accurate".to_string(),
                39_000_000, // ~39MB
                false,
                ModelMetadata::new(
                    "onnx".to_string(),
                    "whisper".to_string(),
                    "multilingual".to_string(),
                    "v1.0".to_string(),
                )
                .with_download_url("https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/model.onnx".to_string())
                .with_quantization("fp16".to_string())
                .with_platform_attribute("webgpu_compatible".to_string(), "true".to_string()),
            ),
            ModelInfo::new(
                "whisper-base".to_string(),
                "Whisper Base".to_string(),
                "Balanced size and accuracy for web usage".to_string(),
                145_000_000, // ~145MB
                false,
                ModelMetadata::new(
                    "onnx".to_string(),
                    "whisper".to_string(),
                    "multilingual".to_string(),
                    "v1.0".to_string(),
                )
                .with_download_url("https://huggingface.co/onnx-community/whisper-base/resolve/main/onnx/model.onnx".to_string())
                .with_quantization("fp16".to_string())
                .with_platform_attribute("webgpu_compatible".to_string(), "true".to_string()),
            ),
            ModelInfo::new(
                "whisper-small".to_string(),
                "Whisper Small".to_string(),
                "Good balance of speed and accuracy".to_string(),
                488_000_000, // ~488MB
                false,
                ModelMetadata::new(
                    "onnx".to_string(),
                    "whisper".to_string(),
                    "multilingual".to_string(),
                    "v1.0".to_string(),
                )
                .with_download_url("https://huggingface.co/onnx-community/whisper-small/resolve/main/onnx/model.onnx".to_string())
                .with_quantization("fp16".to_string())
                .with_platform_attribute("webgpu_compatible".to_string(), "true".to_string()),
            ),
        ];

        for platform_model in default_models {
            // Convert from purr_common::ModelInfo to model::ModelInfo
            let web_model = crate::model::ModelInfo {
                id: platform_model.id,
                name: platform_model.name,
                url: platform_model.metadata.download_url.unwrap_or_default(),
                size: platform_model.size_bytes as usize,
                checksum: None,
                version: platform_model.metadata.version,
                description: platform_model.description,
            };
            self.model_manager.register_model(web_model).await?;
        }

        Ok(())
    }

    /// Check if platform is properly initialized and initialize if needed
    async fn ensure_initialized(&self) -> Result<(), PlatformError> {
        let initialized = self.initialized.read().await;
        if *initialized {
            return Ok(());
        }
        drop(initialized);

        // Use config_mutex to ensure only one initialization happens
        let _guard = self.config_mutex.lock().await;

        // Double-check after acquiring the lock
        let initialized = self.initialized.read().await;
        if *initialized {
            return Ok(());
        }
        drop(initialized);

        // Initialize storage
        self.storage
            .initialize()
            .await
            .map_err(Self::convert_error)?;

        // Initialize default models
        self.initialize_default_models()
            .await
            .map_err(Self::convert_error)?;

        tracing::info!("Platform initialized");

        let mut initialized = self.initialized.write().await;
        *initialized = true;
        Ok(())
    }

    /// Convert `WebError` to `PlatformError`
    fn convert_error(error: WebError) -> PlatformError {
        error.into()
    }
}

#[async_trait::async_trait]
impl Platform for PlatformImpl {
    async fn new() -> Result<Self, PlatformError>
    where
        Self: Sized,
    {
        Self::new().map_err(Self::convert_error)
    }
    // ========================================
    // File Processing Operations
    // ========================================

    async fn process_file(
        &self,
        file_data: Bytes,
        file_path: &Path,
    ) -> Result<FileId, PlatformError> {
        self.ensure_initialized().await?;

        let file_name = file_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");

        // Detect MIME type from file extension
        let mime_type = match file_path.extension().and_then(|ext| ext.to_str()) {
            Some("mp3") => "audio/mpeg",
            Some("wav") => "audio/wav",
            Some("flac") => "audio/flac",
            Some("ogg") => "audio/ogg",
            Some("m4a") => "audio/mp4",
            Some("webm") => "audio/webm",
            _ => "audio/unknown",
        };

        // Store file in browser storage
        let file_id = self
            .storage
            .store_file(file_data, file_name, mime_type)
            .await
            .map_err(Self::convert_error)?;

        Ok(file_id)
    }

    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<TranscriptionStatus, PlatformError>> + Send>>,
        PlatformError,
    > {
        self.ensure_initialized().await?;

        // Get file data
        let file_data = match &request.file {
            FileSource::Bytes(bytes) => bytes.clone(),
            FileSource::Path(path_buf) => panic!(
                "FileSource::Path is not supported in web platform: {}",
                path_buf.display()
            ),
            FileSource::Uploaded(file_id) => self
                .storage
                .get_file(file_id)
                .await
                .map_err(Self::convert_error)?
                .ok_or_else(|| PlatformError::file_processing("File not found".to_string()))?,
        };

        // Create transcription configuration
        let _config = TranscriptionConfig {
            model_name: "whisper-base".to_string(), // Default model
            language: request.language.clone(),
            translate: request.translate,
            ..Default::default()
        };

        // Note: Session management is now handled by start_transcription_process

        // Validate audio file before processing
        validate_audio_file(&file_data, 100 * 1024 * 1024) // 100MB limit
            .map_err(Self::convert_error)?;

        // Use enhanced transcription process with real audio handling
        let transcription_config = TranscriptionConfig {
            model_name: "whisper-base".to_string(),
            language: request.language.clone(),
            translate: request.translate,
            ..Default::default()
        };

        // Configure audio processing with optimized settings
        let audio_config = AudioProcessingConfig {
            target_sample_rate: 16000.0,      // Whisper's optimal sample rate
            target_channels: 1,               // Mono for better transcription
            enable_agc: true,                 // Automatic gain control
            enable_noise_reduction: false,    // Disabled for now to avoid artifacts
            max_file_size: 100 * 1024 * 1024, // 100MB limit
            ..Default::default()
        };

        tracing::info!(
            "Starting enhanced transcription for {} byte file with model: {}",
            file_data.len(),
            transcription_config.model_name
        );

        // Start comprehensive transcription process
        let stream =
            start_transcription_process(file_data, transcription_config, Some(audio_config))
                .await
                .map_err(Self::convert_error)?;

        // Convert to Result stream for platform compatibility
        let converted_stream = stream.map(Ok);

        Ok(Box::pin(converted_stream))
    }

    async fn cleanup(&self, file_id: &str) -> Result<(), PlatformError> {
        self.ensure_initialized().await?;

        let file_id_obj = FileId::from(file_id);

        // Remove file from storage
        self.storage
            .delete_file(&file_id_obj)
            .await
            .map_err(Self::convert_error)?;

        // Close worker session if exists
        {
            let mut sessions = self.sessions.write().await;
            if let Some(session_id) = sessions.remove(file_id) {
                let _ = self.worker.close_session(&session_id).await;
            }
        }

        Ok(())
    }

    // ========================================
    // Model Management Operations
    // ========================================

    async fn list_installed_models(&self) -> Result<Vec<ModelInfo>, PlatformError> {
        self.ensure_initialized().await?;

        let all_models = self
            .model_manager
            .list_models()
            .await
            .map_err(Self::convert_error)?;

        let mut installed_models = Vec::new();
        for model in all_models {
            if self.model_manager.is_model_downloaded(&model.id).await {
                let mut model_info = ModelInfo::new(
                    model.id.clone(),
                    model.name.clone(),
                    model.description.clone(),
                    model.size as u64,
                    true, // installed
                    ModelMetadata::new(
                        "onnx".to_string(),
                        "whisper".to_string(),
                        "multilingual".to_string(),
                        model.version.clone(),
                    )
                    .with_download_url(model.url.clone()),
                );

                // Mark as installed with storage key as path
                model_info
                    .mark_installed(std::path::PathBuf::from(format!("storage://{}", model.id)));
                installed_models.push(model_info);
            }
        }

        Ok(installed_models)
    }

    async fn list_available_models(&self) -> Result<Vec<ModelInfo>, PlatformError> {
        self.ensure_initialized().await?;

        let all_models = self
            .model_manager
            .list_models()
            .await
            .map_err(Self::convert_error)?;

        let available_models = all_models
            .into_iter()
            .map(|model| {
                let is_installed =
                    futures::executor::block_on(self.model_manager.is_model_downloaded(&model.id));

                ModelInfo::new(
                    model.id.clone(),
                    model.name.clone(),
                    model.description.clone(),
                    model.size as u64,
                    is_installed,
                    ModelMetadata::new(
                        "onnx".to_string(),
                        "whisper".to_string(),
                        "multilingual".to_string(),
                        model.version.clone(),
                    )
                    .with_download_url(model.url.clone()),
                )
            })
            .collect();

        Ok(available_models)
    }

    async fn fetch_model(
        &self,
        model_id: &str,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ModelOperationProgress, PlatformError>> + Send>>,
        PlatformError,
    > {
        self.ensure_initialized().await?;

        // Check if model is already downloaded
        if self.model_manager.is_model_downloaded(model_id).await {
            // Return completed immediately
            let progress = vec![ModelOperationProgress::Completed {
                model_id: model_id.to_string(),
                local_path: std::path::PathBuf::from(format!("storage://{model_id}")),
            }];

            let stream = futures::stream::iter(progress.into_iter().map(Ok));
            return Ok(Box::pin(stream));
        }

        // Start download
        let download_stream = self
            .model_manager
            .download_model(model_id)
            .await
            .map_err(Self::convert_error)?;

        // Convert download progress to model operation progress
        let model_id_clone = model_id.to_string();
        let progress_stream = download_stream.map(move |download_progress| {
            if download_progress.percentage == Some(100.0) {
                Ok(ModelOperationProgress::Completed {
                    model_id: model_id_clone.clone(),
                    local_path: std::path::PathBuf::from(format!("storage://{model_id_clone}")),
                })
            } else {
                Ok(ModelOperationProgress::Downloading {
                    model_id: model_id_clone.clone(),
                    bytes_downloaded: download_progress.bytes_downloaded as u64,
                    total_bytes: download_progress.total_bytes.map(|b| b as u64),
                    speed_bps: download_progress.speed_bps.map(|s| s as u64),
                })
            }
        });

        Ok(Box::pin(progress_stream))
    }

    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo, PlatformError> {
        self.ensure_initialized().await?;

        let model = self
            .model_manager
            .get_model_info(model_id)
            .await
            .map_err(Self::convert_error)?
            .ok_or_else(|| PlatformError::model_not_found(model_id))?;

        let is_installed = self.model_manager.is_model_downloaded(model_id).await;

        let mut model_info = ModelInfo::new(
            model.id.clone(),
            model.name.clone(),
            model.description.clone(),
            model.size as u64,
            is_installed,
            ModelMetadata::new(
                "onnx".to_string(),
                "whisper".to_string(),
                "multilingual".to_string(),
                model.version.clone(),
            )
            .with_download_url(model.url.clone()),
        );

        if is_installed {
            model_info.mark_installed(std::path::PathBuf::from(format!("storage://{}", model.id)));
        }

        Ok(model_info)
    }

    async fn remove_model(&self, model_id: &str) -> Result<(), PlatformError> {
        self.ensure_initialized().await?;

        self.model_manager
            .remove_model(model_id)
            .await
            .map_err(Self::convert_error)?;

        Ok(())
    }

    async fn is_model_installed(&self, model_id: &str) -> Result<bool, PlatformError> {
        self.ensure_initialized().await?;

        Ok(self.model_manager.is_model_downloaded(model_id).await)
    }

    async fn get_model_path(&self, model_id: &str) -> Result<String, PlatformError> {
        self.ensure_initialized().await?;

        if !self.model_manager.is_model_downloaded(model_id).await {
            return Err(PlatformError::model_not_found(model_id));
        }

        // For web platform, return storage key
        Ok(format!("storage://{model_id}"))
    }

    async fn list_available_devices(&self) -> Result<Vec<DeviceInfo>, PlatformError> {
        let mut devices = Vec::new();

        // CPU is always available
        devices.push(DeviceInfo {
            id: 0,
            name: "CPU".to_string(),
            description: Some("CPU-based transcription".to_string()),
            device_type: DeviceType::Cpu,
            memory_free: None, // Don't guess values
            memory_total: None,
            capabilities: None,
        });

        // Check if WebGPU is available using message-passing to isolate non-Send operations
        let (result_tx, result_rx) = oneshot::channel();

        // Use spawn_local to isolate all non-Send WebGPU operations
        wasm_bindgen_futures::spawn_local(async move {
            let webgpu_result = async move {
                // Move all window/navigator access inside spawn_local
                let window = match web_sys::window() {
                    Some(w) => w,
                    None => return Err("No window available".to_string()),
                };

                let navigator = window.navigator();

                // Check if WebGPU GPU interface exists
                if navigator.gpu().is_undefined() {
                    return Err("WebGPU not available".to_string());
                }

                let adapter_promise = navigator.gpu().request_adapter();
                let adapter = wasm_bindgen_futures::JsFuture::from(adapter_promise)
                    .await
                    .map_err(|e| format!("Failed to get WebGPU adapter: {e:?}"))?;

                if adapter.is_undefined() {
                    return Err("No WebGPU adapter found".to_string());
                }

                let adapter = GpuAdapter::from(adapter);
                let info = adapter.info();
                let vendor = info.vendor();
                let device = info.device();
                let features = adapter.features();

                let mut caps = HashMap::with_capacity(features.size() as usize);
                let iter = features.keys();
                while let Ok(key) = iter.next() {
                    let key = key.as_string().unwrap_or_default();
                    let value = features.has(&key);
                    caps.insert(key, value.to_string().into());
                }

                Ok::<DeviceInfo, String>(DeviceInfo {
                    id: 1,
                    name: format!("{vendor} {device}"),
                    description: Some("WebGPU-based transcription".to_string()),
                    device_type: DeviceType::Gpu,
                    memory_free: None, // WebGPU does not expose memory info
                    memory_total: None,
                    capabilities: Some(caps),
                })
            };

            let result = webgpu_result.await;
            let _ = result_tx.send(result);
        });

        // Wait for the result from the spawn_local task
        if let Ok(webgpu_result) = result_rx.await {
            match webgpu_result {
                Ok(device_info) => {
                    devices.push(device_info);
                }
                Err(err_msg) => {
                    // Log the error but don't fail - just don't add GPU device
                    tracing::warn!("WebGPU detection failed: {}", err_msg);
                }
            }
        }

        Ok(devices)
    }
}
