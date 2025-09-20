//! Browser-specific model manager using real IndexedDB storage
//! Replaces localStorage simulation with proper persistent storage

use crate::error::{WebError, WebResult};
use crate::indexeddb::{IndexedDBStorage, ModelMetadata, DownloadProgress};
use bytes::Bytes;
use purr_common::platform::FileId;
use purr_core::model::WhisperModel;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use wasm_bindgen::prelude::*;
use web_sys::console;

/// Browser model manager with real IndexedDB persistence
pub struct BrowserModelManager {
    storage: Arc<IndexedDBStorage>,
    download_callbacks: Arc<RwLock<HashMap<String, Box<dyn Fn(DownloadProgress) + Send + Sync>>>>,
}

impl BrowserModelManager {
    /// Create new browser model manager
    pub async fn new() -> WebResult<Self> {
        let storage = Arc::new(IndexedDBStorage::new().await?);

        Ok(Self {
            storage,
            download_callbacks: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Check if a model is already downloaded and complete
    pub async fn is_model_downloaded(&self, model: WhisperModel) -> WebResult<bool> {
        let model_name = model.as_str();
        let models = self.storage.list_models().await?;

        for metadata in models {
            if metadata.model_name == model_name && metadata.is_complete {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Download a Whisper model with real browser APIs
    pub async fn download_model(
        &self,
        model: WhisperModel,
        progress_callback: Option<Box<dyn Fn(DownloadProgress) + Send + Sync>>,
    ) -> WebResult<FileId> {
        let model_name = model.as_str();
        let download_url = Self::get_model_url(model);

        console::log_1(&format!("Starting download of model: {}", model_name).into());

        // Check if already exists and complete
        if self.is_model_downloaded(model).await? {
            return Err(WebError::WebApi {
                message: format!("Model {} already downloaded", model_name),
            });
        }

        // Register progress callback
        if let Some(callback) = progress_callback {
            let mut callbacks = self.download_callbacks.write().await;
            callbacks.insert(model_name.to_string(), callback);
        }

        // Create download callback wrapper
        let callbacks_ref = self.download_callbacks.clone();
        let model_name_clone = model_name.to_string();

        let progress_wrapper = Box::new(move |progress: DownloadProgress| {
            let callbacks = callbacks_ref.clone();
            let model_name = model_name_clone.clone();

            wasm_bindgen_futures::spawn_local(async move {
                let callbacks_guard = callbacks.read().await;
                if let Some(callback) = callbacks_guard.get(&model_name) {
                    callback(progress);
                }
            });
        });

        // Start real download with IndexedDB storage
        let file_id = self.storage.download_model(
            model_name,
            "whisper",
            &download_url,
            Some(progress_wrapper),
        ).await?;

        console::log_1(&format!("Model {} downloaded successfully", model_name).into());

        // Remove callback
        {
            let mut callbacks = self.download_callbacks.write().await;
            callbacks.remove(model_name);
        }

        Ok(file_id)
    }

    /// Get model binary data
    pub async fn get_model_data(&self, model: WhisperModel) -> WebResult<Option<Bytes>> {
        let model_name = model.as_str();
        let models = self.storage.list_models().await?;

        for metadata in models {
            if metadata.model_name == model_name && metadata.is_complete {
                let file_id = FileId::from(metadata.id);
                return self.storage.get_model_binary(&file_id).await;
            }
        }

        Ok(None)
    }

    /// Get model metadata
    pub async fn get_model_metadata(&self, model: WhisperModel) -> WebResult<Option<ModelMetadata>> {
        let model_name = model.as_str();
        let models = self.storage.list_models().await?;

        for metadata in models {
            if metadata.model_name == model_name {
                return Ok(Some(metadata));
            }
        }

        Ok(None)
    }

    /// List all downloaded models
    pub async fn list_downloaded_models(&self) -> WebResult<Vec<WhisperModel>> {
        let models = self.storage.list_models().await?;
        let mut whisper_models = Vec::new();

        for metadata in models {
            if metadata.is_complete {
                if let Ok(model) = metadata.model_name.parse::<WhisperModel>() {
                    whisper_models.push(model);
                }
            }
        }

        Ok(whisper_models)
    }

    /// Delete a downloaded model
    pub async fn delete_model(&self, model: WhisperModel) -> WebResult<bool> {
        let model_name = model.as_str();
        let models = self.storage.list_models().await?;

        for metadata in models {
            if metadata.model_name == model_name {
                let file_id = FileId::from(metadata.id);
                return self.storage.delete_model(&file_id).await;
            }
        }

        Ok(false)
    }

    /// Get storage usage statistics
    pub async fn get_storage_stats(&self) -> WebResult<StorageStats> {
        let (used, total) = self.storage.get_storage_quota().await?;
        let models = self.storage.list_models().await?;

        let mut model_count = 0;
        let mut total_model_size = 0u64;

        for metadata in models {
            if metadata.is_complete {
                model_count += 1;
                total_model_size += metadata.size;
            }
        }

        Ok(StorageStats {
            total_quota: total,
            used_quota: used,
            available_quota: total.saturating_sub(used),
            model_count,
            total_model_size,
            models_percentage: if total > 0 {
                (total_model_size as f64 / total as f64 * 100.0) as u8
            } else {
                0
            },
        })
    }

    /// Request persistent storage permission
    pub async fn request_persistent_storage(&self) -> WebResult<bool> {
        self.storage.request_persistent_storage().await
    }

    /// Clear all stored models
    pub async fn clear_all_models(&self) -> WebResult<()> {
        let models = self.storage.list_models().await?;

        for metadata in models {
            let file_id = FileId::from(metadata.id);
            let _ = self.storage.delete_model(&file_id).await;
        }

        console::log_1(&"All models cleared".into());
        Ok(())
    }

    /// Find the best available model for quick start
    pub async fn find_best_available_model(&self) -> WebResult<Option<WhisperModel>> {
        let downloaded = self.list_downloaded_models().await?;

        // Priority order for best model selection
        let preferred_order = [
            WhisperModel::Base,
            WhisperModel::BaseEn,
            WhisperModel::Small,
            WhisperModel::SmallEn,
            WhisperModel::Tiny,
            WhisperModel::TinyEn,
        ];

        for preferred in preferred_order {
            if downloaded.contains(&preferred) {
                return Ok(Some(preferred));
            }
        }

        // Return any available model
        downloaded.first().copied().map(Some).or(Ok(None))
    }

    /// Get download URL for a Whisper model
    fn get_model_url(model: WhisperModel) -> String {
        let base_url = if model.as_str().contains("tdrz") {
            "https://huggingface.co/akashmjn/tinydiarize-whisper.cpp/resolve/main"
        } else {
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
        };

        format!("{}/ggml-{}.bin", base_url, model.as_str())
    }

    /// Get estimated download time for a model
    pub fn estimate_download_time(&self, model: WhisperModel, speed_mbps: f64) -> f64 {
        let size_mb = model.size() as f64 / (1024.0 * 1024.0);
        let speed_mb_per_sec = speed_mbps / 8.0; // Convert Mbps to MB/s

        if speed_mb_per_sec > 0.0 {
            size_mb / speed_mb_per_sec
        } else {
            f64::INFINITY
        }
    }

    /// Check if storage quota is sufficient for a model
    pub async fn can_download_model(&self, model: WhisperModel) -> WebResult<bool> {
        let stats = self.get_storage_stats().await?;
        let model_size = model.size();

        // Keep 100MB buffer for safety
        let required_space = model_size + (100 * 1024 * 1024);

        Ok(stats.available_quota >= required_space)
    }

    /// Get download progress for a model currently being downloaded
    pub async fn get_download_progress(&self, model: WhisperModel) -> WebResult<Option<u8>> {
        if let Some(metadata) = self.get_model_metadata(model).await? {
            if !metadata.is_complete {
                return Ok(Some(metadata.download_progress));
            }
        }
        Ok(None)
    }
}

/// Storage statistics for the browser
#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_quota: u64,
    pub used_quota: u64,
    pub available_quota: u64,
    pub model_count: usize,
    pub total_model_size: u64,
    pub models_percentage: u8,
}

impl StorageStats {
    /// Check if storage is running low (>80% used)
    pub fn is_storage_low(&self) -> bool {
        if self.total_quota == 0 {
            return false;
        }

        let usage_percentage = (self.used_quota as f64 / self.total_quota as f64) * 100.0;
        usage_percentage > 80.0
    }

    /// Check if storage is critically low (>95% used)
    pub fn is_storage_critical(&self) -> bool {
        if self.total_quota == 0 {
            return false;
        }

        let usage_percentage = (self.used_quota as f64 / self.total_quota as f64) * 100.0;
        usage_percentage > 95.0
    }

    /// Format storage size in human readable format
    pub fn format_size(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", size as u64, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }

    /// Get formatted available space
    pub fn available_space_formatted(&self) -> String {
        Self::format_size(self.available_quota)
    }

    /// Get formatted total space
    pub fn total_space_formatted(&self) -> String {
        Self::format_size(self.total_quota)
    }

    /// Get formatted used space
    pub fn used_space_formatted(&self) -> String {
        Self::format_size(self.used_quota)
    }
}