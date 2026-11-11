//! Web platform implementation of the Platform trait

use crate::error::{WebError, WebResult};
use crate::model::WebModelManager;
use crate::model_cache;
use crate::model_info::model_info;
use crate::storage::WebStorage;
use crate::transcription::{
    AudioProcessingConfig, start_transcription_process, validate_audio_file,
};
use crate::worker::{TranscriptionConfig, TranscriptionWorker, WorkerModel};
use bytes::Bytes;
use futures::StreamExt;
use js_sys::Error as JsError;
use purr_common::platform::{
    DeviceInfo, DeviceType, FileId, FileSource, ModelInfo, ModelProgressStream, Platform,
    PlatformError, TranscriptionRequest, TranscriptionStream,
};
use purr_common::settings::Settings;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, oneshot};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures;
use web_sys::Storage;

pub type PlatformImpl = WasmPlatformImpl;
/// Web platform implementation using WebAssembly and browser APIs
pub struct WasmPlatformImpl {
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

impl WasmPlatformImpl {
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

        tracing::info!("Platform initialized");

        let mut initialized = self.initialized.write().await;
        *initialized = true;
        Ok(())
    }

    /// Convert `WebError` to `PlatformError`
    fn convert_error(error: WebError) -> PlatformError {
        error.into()
    }

    /// Get browser localStorage
    fn get_local_storage() -> Result<Storage, PlatformError> {
        let window = web_sys::window()
            .ok_or_else(|| PlatformError::settings_persistence("No window available".to_owned()))?;
        window
            .local_storage()
            .map_err(|e| {
                PlatformError::settings_persistence(format!("Failed to access localStorage: {e:?}"))
            })?
            .ok_or_else(|| {
                PlatformError::settings_persistence("localStorage not available".to_owned())
            })
    }

    /// Get localStorage value
    fn get_storage_value(key: &str) -> Result<Option<String>, PlatformError> {
        let storage = Self::get_local_storage()?;
        storage.get_item(key).map_err(|e| {
            PlatformError::settings_persistence(format!(
                "Failed to get item '{key}' from localStorage: {e:?}"
            ))
        })
    }

    /// Set localStorage value with quota exceeded handling
    fn set_storage_value(key: &str, value: &str) -> Result<(), PlatformError> {
        let storage = Self::get_local_storage()?;
        storage.set_item(key, value).map_err(|e| {
            // Check if this is a quota exceeded error
            let error_msg = if let Some(js_error) = e.dyn_ref::<JsError>() {
                js_error.message().as_string().unwrap_or_default()
            } else {
                format!("{e:?}")
            };

            if error_msg.to_lowercase().contains("quota")
                || error_msg.to_lowercase().contains("storage")
                || error_msg.contains("22") // DOM_QUOTA_EXCEEDED_ERR
            {
                PlatformError::settings_persistence(format!(
                    "localStorage quota exceeded. Please clear browser storage or reduce settings size: {error_msg}"
                ))
            } else {
                PlatformError::settings_persistence(format!(
                    "Failed to set item '{key}' in localStorage: {error_msg}"
                ))
            }
        })
    }
}

impl WasmPlatformImpl {
    /// Create a new `WebPlatform` instance
    fn new_inner() -> WebResult<Self> {
        let storage = Arc::new(WebStorage::new());
        let model_manager = Arc::new(WebModelManager::new()?);
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

    pub fn new() -> Result<Self, PlatformError>
    where
        Self: Sized,
    {
        Self::new_inner().map_err(Self::convert_error)
    }
}

impl Platform for WasmPlatformImpl {
    fn install_logging_hooks(&self) {
        // No-op for web platform as logging is handled by the host environment
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
    ) -> Result<TranscriptionStream, PlatformError> {
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
                .ok_or_else(|| PlatformError::file_processing("File not found".to_owned()))?,
        };

        // Note: Session management is now handled by start_transcription_process

        // Validate audio file before processing
        validate_audio_file(&file_data, 100 * 1024 * 1024) // 100MB limit
            .map_err(Self::convert_error)?;

        let model = self.model_manager.default_model();
        let transcription_config = TranscriptionConfig {
            language: request.language.clone(),
            translate: request.translate,
            ..TranscriptionConfig::new(WorkerModel::from(model))
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
            "Starting enhanced transcription for {} byte file with model: {}@{}",
            file_data.len(),
            model.repo,
            model.revision
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

        let mut installed = Vec::new();
        for spec in self.model_manager.models() {
            if self
                .model_manager
                .is_model_installed(&spec.id)
                .await
                .map_err(Self::convert_error)?
            {
                installed.push(model_info(spec, true));
            }
        }
        Ok(installed)
    }

    async fn list_available_models(&self) -> Result<Vec<ModelInfo>, PlatformError> {
        self.ensure_initialized().await?;

        let mut available = Vec::with_capacity(self.model_manager.models().len());
        for spec in self.model_manager.models() {
            let is_installed = self
                .model_manager
                .is_model_installed(&spec.id)
                .await
                .map_err(Self::convert_error)?;
            available.push(model_info(spec, is_installed));
        }
        Ok(available)
    }

    async fn fetch_model(&self, model_id: &str) -> Result<ModelProgressStream, PlatformError> {
        self.ensure_initialized().await?;

        self.model_manager
            .model(model_id)
            .map_err(Self::convert_error)?;
        Err(PlatformError::unsupported(
            "fetch_model: transformers.js downloads the pinned files when the model is first used",
        ))
    }

    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo, PlatformError> {
        self.ensure_initialized().await?;

        let spec = self
            .model_manager
            .model(model_id)
            .map_err(Self::convert_error)?;
        let is_installed = self
            .model_manager
            .is_model_installed(model_id)
            .await
            .map_err(Self::convert_error)?;
        Ok(model_info(spec, is_installed))
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

        self.model_manager
            .is_model_installed(model_id)
            .await
            .map_err(Self::convert_error)
    }

    async fn get_model_path(&self, model_id: &str) -> Result<String, PlatformError> {
        self.ensure_initialized().await?;

        let spec = self
            .model_manager
            .model(model_id)
            .map_err(Self::convert_error)?;
        if !self
            .model_manager
            .is_model_installed(model_id)
            .await
            .map_err(Self::convert_error)?
        {
            return Err(PlatformError::model_not_found(model_id));
        }
        Ok(model_cache::location(spec))
    }

    async fn list_available_devices(&self) -> Result<Vec<DeviceInfo>, PlatformError> {
        let mut devices = Vec::new();

        // CPU is always available
        devices.push(DeviceInfo {
            id: 0,
            name: "CPU".to_owned(),
            description: Some("CPU-based transcription".to_owned()),
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
                    None => return Err("No window available".to_owned()),
                };

                let navigator = window.navigator();

                // Check if WebGPU GPU interface exists
                if navigator.gpu().is_undefined() {
                    return Err("WebGPU not available".to_owned());
                }

                let adapter_promise = navigator.gpu().request_adapter();
                let adapter = wasm_bindgen_futures::JsFuture::from(adapter_promise)
                    .await
                    .map_err(|e| format!("Failed to get WebGPU adapter: {e:?}"))?;

                let Some(adapter) = adapter.into_option() else {
                    return Err("No WebGPU adapter found".to_owned());
                };
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
                    description: Some("WebGPU-based transcription".to_owned()),
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

    // ========================================
    // Settings Persistence Operations
    // ========================================

    async fn load_settings(&self) -> Result<Settings, PlatformError> {
        const SETTINGS_KEY: &str = "purr-settings";

        if let Some(json_str) = Self::get_storage_value(SETTINGS_KEY)? {
            // Try to parse stored settings
            match serde_json::from_str(&json_str) {
                Ok(settings) => Ok(settings),
                Err(e) => {
                    tracing::warn!("Failed to parse stored settings, using defaults: {}", e);
                    // If parsing fails, fall back to default settings
                    // This provides automatic recovery from corrupted settings
                    Ok(Settings::default())
                }
            }
        } else {
            // No settings found, return default settings
            tracing::info!("No settings found in localStorage, using defaults");
            Ok(Settings::default())
        }
    }

    async fn save_settings(&self, settings: &Settings) -> Result<(), PlatformError> {
        const SETTINGS_KEY: &str = "purr-settings";

        // Validate settings before saving
        let validation_errors = settings.validate();
        if !validation_errors.is_empty() {
            let error_messages: Vec<String> = validation_errors
                .into_iter()
                .map(|e| e.to_string())
                .collect();
            return Err(PlatformError::settings_persistence(format!(
                "Invalid settings: {}",
                error_messages.join(", ")
            )));
        }

        // Serialize settings to JSON
        let json_str = serde_json::to_string_pretty(settings).map_err(|e| {
            PlatformError::settings_persistence(format!("Failed to serialize settings: {e}"))
        })?;

        // Save to localStorage with quota handling
        Self::set_storage_value(SETTINGS_KEY, &json_str)?;

        tracing::info!("Settings saved to localStorage successfully");
        Ok(())
    }

    async fn get_settings_location(&self) -> Result<Option<String>, PlatformError> {
        Ok(Some("localStorage:purr-settings".to_owned()))
    }

    // ========================================
    // Theme Storage Implementation
    // ========================================

    async fn load_theme_setting(&self, storage_key: &str) -> Result<Option<String>, PlatformError> {
        // Load theme from localStorage
        Self::get_storage_value(storage_key)
    }

    async fn save_theme_setting(
        &self,
        storage_key: &str,
        theme: &str,
    ) -> Result<(), PlatformError> {
        // Save theme to localStorage
        Self::set_storage_value(storage_key, theme)
    }
}
