//! IndexedDB integration for web workers
//! Provides real model storage access from web workers

use crate::error::{WebError, WebResult};
use crate::indexeddb::{IndexedDBStorage, ModelMetadata};
use bytes::Bytes;
use purr_common::platform::FileId;
use purr_core::model::WhisperModel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::*;

/// Worker-side model cache with IndexedDB backend
pub struct WorkerModelCache {
    storage: Arc<IndexedDBStorage>,
    memory_cache: Arc<RwLock<HashMap<String, Bytes>>>,
    metadata_cache: Arc<RwLock<HashMap<String, ModelMetadata>>>,
}

impl WorkerModelCache {
    /// Create new worker model cache
    pub async fn new() -> WebResult<Self> {
        let storage = Arc::new(IndexedDBStorage::new().await?);

        Ok(Self {
            storage,
            memory_cache: Arc::new(RwLock::new(HashMap::new())),
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Load model data for worker use
    pub async fn load_model(&self, model: WhisperModel) -> WebResult<Option<Bytes>> {
        let model_name = model.as_str();

        // Check memory cache first
        {
            let cache = self.memory_cache.read().await;
            if let Some(data) = cache.get(model_name) {
                return Ok(Some(data.clone()));
            }
        }

        // Load from IndexedDB
        let models = self.storage.list_models().await?;
        for metadata in models {
            if metadata.model_name == model_name && metadata.is_complete {
                let file_id = FileId::from(metadata.id.clone());
                if let Some(data) = self.storage.get_model_binary(&file_id).await? {
                    // Cache in memory for faster access
                    {
                        let mut cache = self.memory_cache.write().await;
                        cache.insert(model_name.to_string(), data.clone());
                    }
                    {
                        let mut meta_cache = self.metadata_cache.write().await;
                        meta_cache.insert(model_name.to_string(), metadata);
                    }

                    return Ok(Some(data));
                }
            }
        }

        Ok(None)
    }

    /// Check if model is available
    pub async fn is_model_available(&self, model: WhisperModel) -> WebResult<bool> {
        let model_name = model.as_str();

        // Check memory cache
        {
            let cache = self.memory_cache.read().await;
            if cache.contains_key(model_name) {
                return Ok(true);
            }
        }

        // Check IndexedDB
        let models = self.storage.list_models().await?;
        for metadata in models {
            if metadata.model_name == model_name && metadata.is_complete {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get model metadata
    pub async fn get_model_metadata(&self, model: WhisperModel) -> WebResult<Option<ModelMetadata>> {
        let model_name = model.as_str();

        // Check cache first
        {
            let cache = self.metadata_cache.read().await;
            if let Some(metadata) = cache.get(model_name) {
                return Ok(Some(metadata.clone()));
            }
        }

        // Load from IndexedDB
        let models = self.storage.list_models().await?;
        for metadata in models {
            if metadata.model_name == model_name {
                // Cache for future use
                {
                    let mut cache = self.metadata_cache.write().await;
                    cache.insert(model_name.to_string(), metadata.clone());
                }
                return Ok(Some(metadata));
            }
        }

        Ok(None)
    }

    /// Clear memory cache to free RAM
    pub async fn clear_memory_cache(&self) {
        let mut cache = self.memory_cache.write().await;
        cache.clear();
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> WebResult<CacheStats> {
        let memory_cache = self.memory_cache.read().await;
        let metadata_cache = self.metadata_cache.read().await;

        let memory_size = memory_cache
            .values()
            .map(|data| data.len() as u64)
            .sum();

        Ok(CacheStats {
            memory_cache_count: memory_cache.len(),
            metadata_cache_count: metadata_cache.len(),
            memory_cache_size: memory_size,
        })
    }

    /// Preload commonly used models into memory
    pub async fn preload_models(&self, models: &[WhisperModel]) -> WebResult<Vec<String>> {
        let mut loaded = Vec::new();

        for model in models {
            match self.load_model(*model).await {
                Ok(Some(_)) => {
                    loaded.push(model.as_str().to_string());
                    console::log_1(&format!("Preloaded model: {}", model.as_str()).into());
                }
                Ok(None) => {
                    console::log_1(&format!("Model not available: {}", model.as_str()).into());
                }
                Err(e) => {
                    console::error_1(&format!("Failed to preload {}: {:?}", model.as_str(), e).into());
                }
            }
        }

        Ok(loaded)
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub memory_cache_count: usize,
    pub metadata_cache_count: usize,
    pub memory_cache_size: u64,
}

/// Worker message types for model operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerModelMessage {
    LoadModel {
        model_name: String,
        request_id: String,
    },
    CheckModelAvailable {
        model_name: String,
        request_id: String,
    },
    GetModelMetadata {
        model_name: String,
        request_id: String,
    },
    PreloadModels {
        model_names: Vec<String>,
        request_id: String,
    },
    ClearCache {
        request_id: String,
    },
    GetCacheStats {
        request_id: String,
    },
}

/// Worker response types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerModelResponse {
    ModelLoaded {
        request_id: String,
        success: bool,
        size: Option<u64>,
        error: Option<String>,
    },
    ModelAvailable {
        request_id: String,
        available: bool,
    },
    ModelMetadata {
        request_id: String,
        metadata: Option<ModelMetadata>,
    },
    ModelsPreloaded {
        request_id: String,
        loaded_models: Vec<String>,
    },
    CacheCleared {
        request_id: String,
    },
    CacheStats {
        request_id: String,
        stats: CacheStats,
    },
    Error {
        request_id: String,
        error: String,
    },
}

/// Handle worker model messages
pub async fn handle_model_message(
    cache: &WorkerModelCache,
    message: WorkerModelMessage,
) -> WorkerModelResponse {
    match message {
        WorkerModelMessage::LoadModel { model_name, request_id } => {
            match model_name.parse::<WhisperModel>() {
                Ok(model) => {
                    match cache.load_model(model).await {
                        Ok(Some(data)) => WorkerModelResponse::ModelLoaded {
                            request_id,
                            success: true,
                            size: Some(data.len() as u64),
                            error: None,
                        },
                        Ok(None) => WorkerModelResponse::ModelLoaded {
                            request_id,
                            success: false,
                            size: None,
                            error: Some("Model not found".to_string()),
                        },
                        Err(e) => WorkerModelResponse::Error {
                            request_id,
                            error: format!("Failed to load model: {:?}", e),
                        },
                    }
                }
                Err(e) => WorkerModelResponse::Error {
                    request_id,
                    error: format!("Invalid model name: {:?}", e),
                },
            }
        }

        WorkerModelMessage::CheckModelAvailable { model_name, request_id } => {
            match model_name.parse::<WhisperModel>() {
                Ok(model) => {
                    match cache.is_model_available(model).await {
                        Ok(available) => WorkerModelResponse::ModelAvailable {
                            request_id,
                            available,
                        },
                        Err(e) => WorkerModelResponse::Error {
                            request_id,
                            error: format!("Failed to check model availability: {:?}", e),
                        },
                    }
                }
                Err(e) => WorkerModelResponse::Error {
                    request_id,
                    error: format!("Invalid model name: {:?}", e),
                },
            }
        }

        WorkerModelMessage::GetModelMetadata { model_name, request_id } => {
            match model_name.parse::<WhisperModel>() {
                Ok(model) => {
                    match cache.get_model_metadata(model).await {
                        Ok(metadata) => WorkerModelResponse::ModelMetadata {
                            request_id,
                            metadata,
                        },
                        Err(e) => WorkerModelResponse::Error {
                            request_id,
                            error: format!("Failed to get metadata: {:?}", e),
                        },
                    }
                }
                Err(e) => WorkerModelResponse::Error {
                    request_id,
                    error: format!("Invalid model name: {:?}", e),
                },
            }
        }

        WorkerModelMessage::PreloadModels { model_names, request_id } => {
            let models: Vec<WhisperModel> = model_names
                .iter()
                .filter_map(|name| name.parse().ok())
                .collect();

            match cache.preload_models(&models).await {
                Ok(loaded_models) => WorkerModelResponse::ModelsPreloaded {
                    request_id,
                    loaded_models,
                },
                Err(e) => WorkerModelResponse::Error {
                    request_id,
                    error: format!("Failed to preload models: {:?}", e),
                },
            }
        }

        WorkerModelMessage::ClearCache { request_id } => {
            cache.clear_memory_cache().await;
            WorkerModelResponse::CacheCleared { request_id }
        }

        WorkerModelMessage::GetCacheStats { request_id } => {
            match cache.get_cache_stats().await {
                Ok(stats) => WorkerModelResponse::CacheStats { request_id, stats },
                Err(e) => WorkerModelResponse::Error {
                    request_id,
                    error: format!("Failed to get cache stats: {:?}", e),
                },
            }
        }
    }
}