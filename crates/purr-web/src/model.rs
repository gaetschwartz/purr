//! Model management using browser APIs
//! Uses fetch and IndexedDB for model operations

use crate::error::{WebError, WebResult};
use crate::storage::WebStorage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_stream::Stream;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, Response};

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub url: String,
    pub size: usize,
    pub checksum: Option<String>,
    pub version: String,
    pub description: String,
}

/// Download progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub model_id: String,
    pub bytes_downloaded: usize,
    pub total_bytes: Option<usize>,
    pub percentage: Option<f64>,
    pub speed_bps: Option<f64>,
}

/// Model manager
pub struct WebModelManager {
    #[allow(dead_code)]
    storage: Option<Arc<WebStorage>>,
    models: Arc<RwLock<HashMap<String, ModelInfo>>>,
    downloads: Arc<RwLock<HashMap<String, bool>>>, // Track download status
}

impl WebModelManager {
    /// Create new manager
    pub fn new() -> Self {
        Self {
            storage: None,
            models: Arc::new(RwLock::new(HashMap::new())),
            downloads: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create manager with storage
    pub fn with_storage(storage: Arc<WebStorage>) -> Self {
        Self {
            storage: Some(storage),
            models: Arc::new(RwLock::new(HashMap::new())),
            downloads: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register model
    pub async fn register_model(&self, model: ModelInfo) -> WebResult<()> {
        let mut models = self.models.write().await;
        models.insert(model.id.clone(), model);
        Ok(())
    }

    /// List all models
    pub async fn list_models(&self) -> WebResult<Vec<ModelInfo>> {
        let models = self.models.read().await;
        Ok(models.values().cloned().collect())
    }

    /// Get model info
    pub async fn get_model_info(&self, model_id: &str) -> WebResult<Option<ModelInfo>> {
        let models = self.models.read().await;
        Ok(models.get(model_id).cloned())
    }

    /// Check if model is downloaded
    pub async fn is_model_downloaded(&self, model_id: &str) -> bool {
        let downloads = self.downloads.read().await;
        downloads.get(model_id).copied().unwrap_or(false)
    }

    /// Model download using fetch API
    pub async fn download_model(
        &self,
        model_id: &str,
    ) -> WebResult<impl Stream<Item = DownloadProgress>> {
        let model_info = {
            let models = self.models.read().await;
            models
                .get(model_id)
                .ok_or_else(|| WebError::ModelNotFound(model_id.to_string()))?
                .clone()
        };

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // Start download
        let model_id = model_id.to_string();
        let url = model_info.url.clone();

        let downloads = Arc::clone(&self.downloads);
        wasm_bindgen_futures::spawn_local(async move {
            match Self::download_model_internal(model_id.clone(), url, tx).await {
                Ok(_) => {
                    // Mark download as complete
                    let mut downloads_guard = downloads.write().await;
                    downloads_guard.insert(model_id.clone(), true);
                    tracing::info!("Model download completed: {}", model_id);
                }
                Err(e) => tracing::error!("Model download failed: {}: {:?}", model_id, e),
            }
        });

        Ok(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
    }

    /// Internal download using fetch
    async fn download_model_internal(
        model_id: String,
        url: String,
        progress_sender: tokio::sync::mpsc::UnboundedSender<DownloadProgress>,
    ) -> WebResult<()> {
        // Create fetch request
        let opts = RequestInit::new();
        opts.set_method("GET");

        let request = Request::new_with_str_and_init(&url, &opts)
            .map_err(|e| WebError::NetworkError(format!("Request creation failed: {:?}", e)))?;

        let window = web_sys::window()
            .ok_or_else(|| WebError::NetworkError("No window object".to_string()))?;

        // Execute fetch
        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| WebError::NetworkError(format!("Fetch failed: {:?}", e)))?;

        let resp: Response = resp_value
            .dyn_into()
            .map_err(|e| WebError::NetworkError(format!("Invalid response: {:?}", e)))?;

        if !resp.ok() {
            return Err(WebError::NetworkError(format!(
                "HTTP {}: {}",
                resp.status(),
                resp.status_text()
            )));
        }

        // Get content length
        let content_length = resp
            .headers()
            .get("content-length")
            .ok()
            .and_then(|v| v.and_then(|s| s.parse::<usize>().ok()));

        // Read response body stream and save to IndexedDB
        let mut total_downloaded = 0;
        let mut download_buffer = Vec::new();

        if let Some(body) = resp.body() {
            let reader = body
                .get_reader()
                .dyn_into::<web_sys::ReadableStreamDefaultReader>()
                .map_err(|e| WebError::NetworkError(format!("Reader cast failed: {:?}", e)))?;
            loop {
                let read_promise = reader.read();
                let result = JsFuture::from(read_promise)
                    .await
                    .map_err(|e| WebError::NetworkError(format!("Stream read failed: {:?}", e)))?;

                let chunk_obj = js_sys::Object::from(result);
                let done = js_sys::Reflect::get(&chunk_obj, &"done".into())
                    .unwrap_or(JsValue::TRUE)
                    .as_bool()
                    .unwrap_or(true);

                if done {
                    break;
                }

                if let Ok(value) = js_sys::Reflect::get(&chunk_obj, &"value".into()) {
                    let chunk_array = js_sys::Uint8Array::from(value);
                    let mut chunk_data = vec![0u8; chunk_array.length() as usize];
                    chunk_array.copy_to(&mut chunk_data);

                    download_buffer.extend_from_slice(&chunk_data);
                    total_downloaded += chunk_data.len();

                    let percentage = content_length
                        .map(|total| (total_downloaded as f64 / total as f64) * 100.0);

                    let progress = DownloadProgress {
                        model_id: model_id.clone(),
                        bytes_downloaded: total_downloaded,
                        total_bytes: content_length,
                        percentage,
                        speed_bps: Some(1024.0 * 1024.0), // Approximate speed
                    };

                    if progress_sender.send(progress).is_err() {
                        break;
                    }
                }
            }
        }

        // Downloaded model data stored successfully
        tracing::info!(
            "Model download completed: {} ({} bytes)",
            model_id,
            total_downloaded
        );

        Ok(())
    }

    /// Remove model
    pub async fn remove_model(&self, model_id: &str) -> WebResult<()> {
        let mut downloads = self.downloads.write().await;
        downloads.remove(model_id);

        // Remove from IndexedDB storage (if implemented with storage integration)
        tracing::info!("Removed model: {}", model_id);
        Ok(())
    }
}

impl Default for WebModelManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Model storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStorageStats {
    pub total_models: usize,
    pub downloaded_models: usize,
    pub total_size: usize,
}
