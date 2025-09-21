//! Real IndexedDB-based storage for browser model persistence
//! Implements actual browser storage using IndexedDB for model binaries and metadata

use crate::error::{WebError, WebResult};
use bytes::Bytes;
use js_sys::{Array, Promise, Uint8Array};
use purr_common::platform::FileId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::*;

/// Model metadata for IndexedDB storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub model_name: String,
    pub model_type: String,
    pub file_name: String,
    pub size: u64,
    pub download_url: String,
    pub created_at: f64,
    pub last_accessed: f64,
    pub download_progress: u8, // 0-100
    pub is_complete: bool,
    pub sha256_hash: Option<String>,
}

/// Download progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub downloaded: u64,
    pub total: u64,
    pub speed_bytes_per_sec: u64,
    pub elapsed_ms: u64,
    pub eta_ms: Option<u64>,
}

/// Real IndexedDB storage manager for models
pub struct IndexedDBStorage {
    db_name: String,
    db_version: u32,
    db: Option<IdbDatabase>,
}

impl IndexedDBStorage {
    pub const DB_NAME: &'static str = "PurrWebGPUModels";
    pub const DB_VERSION: u32 = 1;

    // Object store names
    pub const MODELS_STORE: &'static str = "models";
    pub const METADATA_STORE: &'static str = "metadata";
    pub const DOWNLOADS_STORE: &'static str = "downloads";

    /// Create new IndexedDB storage manager
    pub async fn new() -> WebResult<Self> {
        let mut storage = Self {
            db_name: Self::DB_NAME.to_string(),
            db_version: Self::DB_VERSION,
            db: None,
        };

        storage.init_database().await?;
        Ok(storage)
    }

    /// Initialize IndexedDB database with proper schema
    async fn init_database(&mut self) -> WebResult<()> {
        let window = web_sys::window().ok_or_else(|| WebError::WebApi {
            api: "Window".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::NotFound, "Window not available")),
        })?;

        let idb_factory = window
            .indexed_db()
            .map_err(|e| WebError::web_api_js("IndexedDB", e))?
            .ok_or_else(|| WebError::WebApi {
                api: "IndexedDB".to_string(),
                source: Box::new(std::io::Error::new(std::io::ErrorKind::NotFound, "IndexedDB not supported")),
            })?;

        // Open database with version
        let open_request = idb_factory
            .open_with_u32(&self.db_name, self.db_version)
            .map_err(|e| WebError::web_api_js("IndexedDB open", e))?;

        // Set up upgrade handler
        let upgrade_closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
            let target = event.target().unwrap();
            let request: IdbOpenDbRequest = target.dyn_into().unwrap();
            let db: IdbDatabase = request.result().unwrap().dyn_into().unwrap();

            // Create object stores
            Self::create_object_stores(&db).unwrap();
        }) as Box<dyn Fn(web_sys::Event)>);

        open_request.set_onupgradeneeded(Some(upgrade_closure.as_ref().unchecked_ref()));

        // Wait for database to open
        let db_promise = Promise::new(&mut |resolve, reject| {
            let success_closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
                let target = event.target().unwrap();
                let request: IdbOpenDbRequest = target.dyn_into().unwrap();
                let db = request.result().unwrap();
                resolve.call1(&JsValue::undefined(), &db).unwrap();
            }) as Box<dyn Fn(web_sys::Event)>);

            let error_closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
                let target = event.target().unwrap();
                let request: IdbOpenDbRequest = target.dyn_into().unwrap();
                let error = request.error().unwrap();
                reject.call1(&JsValue::undefined(), &error).unwrap();
            }) as Box<dyn Fn(web_sys::Event)>);

            open_request.set_onsuccess(Some(success_closure.as_ref().unchecked_ref()));
            open_request.set_onerror(Some(error_closure.as_ref().unchecked_ref()));

            // Keep closures alive
            success_closure.forget();
            error_closure.forget();
        });

        let db_result = JsFuture::from(db_promise).await
            .map_err(|e| WebError::web_api_js("Database initialization", e))?;

        self.db = Some(db_result.dyn_into::<IdbDatabase>().unwrap());
        upgrade_closure.forget();

        Ok(())
    }

    /// Create IndexedDB object stores with proper indexes
    fn create_object_stores(db: &IdbDatabase) -> WebResult<()> {
        // Models store for binary data
        if !db.object_store_names().contains(Self::MODELS_STORE) {
            let models_store = db
                .create_object_store(Self::MODELS_STORE)
                .map_err(|e| WebError::web_api_js("Create models store", e))?;

            // Index by model name
            models_store
                .create_index("by_model_name", &JsValue::from_str("model_name"))
                .map_err(|e| WebError::web_api_js("Create model name index", e))?;
        }

        // Metadata store for model information
        if !db.object_store_names().contains(Self::METADATA_STORE) {
            let metadata_store = db
                .create_object_store(Self::METADATA_STORE)
                .map_err(|e| WebError::web_api_js("Create metadata store", e))?;

            // Index by model type
            metadata_store
                .create_index("by_type", &JsValue::from_str("model_type"))
                .map_err(|e| WebError::web_api_js("Create type index", e))?;

            // Index by completion status
            metadata_store
                .create_index("by_complete", &JsValue::from_str("is_complete"))
                .map_err(|e| WebError::web_api_js("Create completion index", e))?;
        }

        // Downloads store for progress tracking
        if !db.object_store_names().contains(Self::DOWNLOADS_STORE) {
            db.create_object_store(Self::DOWNLOADS_STORE)
                .map_err(|e| WebError::web_api_js("Create downloads store", e))?
        }

        Ok(())
    }

    /// Download a model with real fetch API and progress tracking
    pub async fn download_model(
        &self,
        model_name: &str,
        model_type: &str,
        download_url: &str,
        progress_callback: Option<Box<dyn Fn(DownloadProgress)>>,
    ) -> WebResult<FileId> {
        let start_time = js_sys::Date::now();
        let file_id = uuid::Uuid::new_v4().to_string();

        // Create initial metadata
        let metadata = ModelMetadata {
            id: file_id.clone(),
            model_name: model_name.to_string(),
            model_type: model_type.to_string(),
            file_name: format!("{}.bin", model_name),
            size: 0,
            download_url: download_url.to_string(),
            created_at: start_time,
            last_accessed: start_time,
            download_progress: 0,
            is_complete: false,
            sha256_hash: None,
        };

        // Store initial metadata
        self.store_metadata(&metadata).await?;

        // Start download with fetch API
        let window = web_sys::window().ok_or_else(|| WebError::WebApi {
            api: "Window".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::NotFound, "Window not available")),
        })?;

        let response = JsFuture::from(window.fetch_with_str(download_url))
            .await
            .map_err(|e| WebError::web_api_js("Fetch", e))?;

        let response: Response = response.dyn_into().unwrap();

        if !response.ok() {
            return Err(WebError::WebApi {
                message: format!("HTTP {} - {}", response.status(), response.status_text()),
            });
        }

        // Get content length for progress
        let content_length = response
            .headers()
            .get("content-length")
            .unwrap_or(None)
            .and_then(|cl| cl.parse::<u64>().ok())
            .unwrap_or(0);

        // Update metadata with total size
        let mut updated_metadata = metadata.clone();
        updated_metadata.size = content_length;
        self.store_metadata(&updated_metadata).await?;

        // Get readable stream
        let body = response.body().ok_or_else(|| WebError::WebApi {
            api: "Response body".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::NotFound, "Response body not available")),
        })?;

        let reader = body.get_reader();

        // Download with progress tracking
        let mut downloaded = 0u64;
        let mut chunks = Vec::new();

        loop {
            let read_promise = reader.read();
            let read_result = JsFuture::from(read_promise).await
                .map_err(|e| WebError::WebApi {
                    message: format!("Read failed: {:?}", e),
                })?;

            let read_object = js_sys::Object::from(read_result);
            let done = js_sys::Reflect::get(&read_object, &JsValue::from_str("done"))
                .unwrap()
                .as_bool()
                .unwrap_or(false);

            if done {
                break;
            }

            let value = js_sys::Reflect::get(&read_object, &JsValue::from_str("value"))
                .unwrap();

            if !value.is_undefined() {
                let uint8_array: Uint8Array = value.dyn_into().unwrap();
                let chunk_data = uint8_array.to_vec();
                downloaded += chunk_data.len() as u64;
                chunks.extend_from_slice(&chunk_data);

                // Calculate progress
                let elapsed_ms = js_sys::Date::now() - start_time;
                let progress_percent = if content_length > 0 {
                    ((downloaded as f64 / content_length as f64) * 100.0) as u8
                } else {
                    0
                };

                let speed = if elapsed_ms > 0.0 {
                    (downloaded as f64 / (elapsed_ms / 1000.0)) as u64
                } else {
                    0
                };

                let eta = if speed > 0 && content_length > downloaded {
                    Some(((content_length - downloaded) as f64 / speed as f64 * 1000.0) as u64)
                } else {
                    None
                };

                let progress = DownloadProgress {
                    downloaded,
                    total: content_length,
                    speed_bytes_per_sec: speed,
                    elapsed_ms: elapsed_ms as u64,
                    eta_ms: eta,
                };

                // Update metadata progress
                updated_metadata.download_progress = progress_percent;
                self.store_metadata(&updated_metadata).await?;

                // Store progress
                self.store_download_progress(&file_id, &progress).await?;

                // Call progress callback
                if let Some(ref callback) = progress_callback {
                    callback(progress);
                }
            }
        }

        // Store the complete model binary
        let model_data = Bytes::from(chunks);
        self.store_model_binary(&file_id, model_data).await?;

        // Mark as complete
        updated_metadata.is_complete = true;
        updated_metadata.download_progress = 100;
        updated_metadata.size = downloaded;
        self.store_metadata(&updated_metadata).await?;

        Ok(FileId::from(file_id))
    }

    /// Store model binary data in IndexedDB
    async fn store_model_binary(&self, file_id: &str, data: Bytes) -> WebResult<()> {
        let db = self.db.as_ref().ok_or_else(|| WebError::WebApi {
            api: "Database".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::NotConnected, "Database not initialized")),
        })?;

        let transaction = db
            .transaction_with_str_and_mode(Self::MODELS_STORE, IdbTransactionMode::Readwrite)
            .map_err(|e| WebError::web_api_js("Create transaction", e))?;

        let store = transaction
            .object_store(Self::MODELS_STORE)
            .map_err(|e| WebError::web_api_js("Get object store", e))?;

        // Convert Bytes to Uint8Array for storage
        let uint8_array = Uint8Array::new_with_length(data.len() as u32);
        uint8_array.copy_from(&data);

        let request = store
            .put_with_key(&uint8_array.into(), &JsValue::from_str(file_id))
            .map_err(|e| WebError::web_api_js("Store model", e))?;

        // Wait for completion
        let promise = Promise::new(&mut |resolve, reject| {
            let success_closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                resolve.call0(&JsValue::undefined()).unwrap();
            }) as Box<dyn Fn(web_sys::Event)>);

            let error_closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
                let target = event.target().unwrap();
                let request: IdbRequest = target.dyn_into().unwrap();
                let error = request.error().unwrap();
                reject.call1(&JsValue::undefined(), &error).unwrap();
            }) as Box<dyn Fn(web_sys::Event)>);

            request.set_onsuccess(Some(success_closure.as_ref().unchecked_ref()));
            request.set_onerror(Some(error_closure.as_ref().unchecked_ref()));

            success_closure.forget();
            error_closure.forget();
        });

        JsFuture::from(promise).await
            .map_err(|e| WebError::web_api_js("Model storage", e))?;

        Ok(())
    }

    /// Store model metadata
    async fn store_metadata(&self, metadata: &ModelMetadata) -> WebResult<()> {
        let db = self.db.as_ref().ok_or_else(|| WebError::WebApi {
            api: "Database".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::NotConnected, "Database not initialized")),
        })?;

        let transaction = db
            .transaction_with_str_and_mode(Self::METADATA_STORE, IdbTransactionMode::Readwrite)
            .map_err(|e| WebError::web_api_js("Create transaction", e))?;

        let store = transaction
            .object_store(Self::METADATA_STORE)
            .map_err(|e| WebError::web_api_js("Get object store", e))?;

        // Serialize metadata to JSON
        let metadata_json = serde_json::to_string(metadata)
            .map_err(|e| WebError::WebApi {
                message: format!("Failed to serialize metadata: {}", e),
            })?;

        let request = store
            .put_with_key(&JsValue::from_str(&metadata_json), &JsValue::from_str(&metadata.id))
            .map_err(|e| WebError::web_api_js("Store metadata", e))?;

        // Wait for completion
        let promise = Promise::new(&mut |resolve, reject| {
            let success_closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                resolve.call0(&JsValue::undefined()).unwrap();
            }) as Box<dyn Fn(web_sys::Event)>);

            let error_closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
                let target = event.target().unwrap();
                let request: IdbRequest = target.dyn_into().unwrap();
                let error = request.error().unwrap();
                reject.call1(&JsValue::undefined(), &error).unwrap();
            }) as Box<dyn Fn(web_sys::Event)>);

            request.set_onsuccess(Some(success_closure.as_ref().unchecked_ref()));
            request.set_onerror(Some(error_closure.as_ref().unchecked_ref()));

            success_closure.forget();
            error_closure.forget();
        });

        JsFuture::from(promise).await
            .map_err(|e| WebError::web_api_js("Metadata storage", e))?;

        Ok(())
    }

    /// Store download progress
    async fn store_download_progress(&self, file_id: &str, progress: &DownloadProgress) -> WebResult<()> {
        let db = self.db.as_ref().ok_or_else(|| WebError::WebApi {
            api: "Database".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::NotConnected, "Database not initialized")),
        })?;

        let transaction = db
            .transaction_with_str_and_mode(Self::DOWNLOADS_STORE, IdbTransactionMode::Readwrite)
            .map_err(|e| WebError::web_api_js("Create transaction", e))?;

        let store = transaction
            .object_store(Self::DOWNLOADS_STORE)
            .map_err(|e| WebError::web_api_js("Get object store", e))?;

        let progress_json = serde_json::to_string(progress)
            .map_err(|e| WebError::WebApi {
                message: format!("Failed to serialize progress: {}", e),
            })?;

        store
            .put_with_key(&JsValue::from_str(&progress_json), &JsValue::from_str(file_id))
            .map_err(|e| WebError::web_api_js("Store progress", e))?;

        Ok(())
    }

    /// Retrieve model binary data
    pub async fn get_model_binary(&self, file_id: &FileId) -> WebResult<Option<Bytes>> {
        let db = self.db.as_ref().ok_or_else(|| WebError::WebApi {
            api: "Database".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::NotConnected, "Database not initialized")),
        })?;

        let transaction = db
            .transaction_with_str(Self::MODELS_STORE)
            .map_err(|e| WebError::web_api_js("Create transaction", e))?;

        let store = transaction
            .object_store(Self::MODELS_STORE)
            .map_err(|e| WebError::web_api_js("Get object store", e))?;

        let request = store
            .get(&JsValue::from_str(&file_id.to_string()))
            .map_err(|e| WebError::web_api_js("Get model", e))?;

        let promise = Promise::new(&mut |resolve, reject| {
            let success_closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
                let target = event.target().unwrap();
                let request: IdbRequest = target.dyn_into().unwrap();
                let result = request.result().unwrap();
                resolve.call1(&JsValue::undefined(), &result).unwrap();
            }) as Box<dyn Fn(web_sys::Event)>);

            let error_closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
                let target = event.target().unwrap();
                let request: IdbRequest = target.dyn_into().unwrap();
                let error = request.error().unwrap();
                reject.call1(&JsValue::undefined(), &error).unwrap();
            }) as Box<dyn Fn(web_sys::Event)>);

            request.set_onsuccess(Some(success_closure.as_ref().unchecked_ref()));
            request.set_onerror(Some(error_closure.as_ref().unchecked_ref()));

            success_closure.forget();
            error_closure.forget();
        });

        let result = JsFuture::from(promise).await
            .map_err(|e| WebError::web_api_js("Model retrieval", e))?;

        if result.is_undefined() {
            return Ok(None);
        }

        let uint8_array: Uint8Array = result.dyn_into().unwrap();
        let data = uint8_array.to_vec();
        Ok(Some(Bytes::from(data)))
    }

    /// Get model metadata
    pub async fn get_metadata(&self, file_id: &FileId) -> WebResult<Option<ModelMetadata>> {
        let db = self.db.as_ref().ok_or_else(|| WebError::WebApi {
            api: "Database".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::NotConnected, "Database not initialized")),
        })?;

        let transaction = db
            .transaction_with_str(Self::METADATA_STORE)
            .map_err(|e| WebError::web_api_js("Create transaction", e))?;

        let store = transaction
            .object_store(Self::METADATA_STORE)
            .map_err(|e| WebError::web_api_js("Get object store", e))?;

        let request = store
            .get(&JsValue::from_str(&file_id.to_string()))
            .map_err(|e| WebError::web_api_js("Get metadata", e))?;

        let promise = Promise::new(&mut |resolve, reject| {
            let success_closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
                let target = event.target().unwrap();
                let request: IdbRequest = target.dyn_into().unwrap();
                let result = request.result().unwrap();
                resolve.call1(&JsValue::undefined(), &result).unwrap();
            }) as Box<dyn Fn(web_sys::Event)>);

            let error_closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
                let target = event.target().unwrap();
                let request: IdbRequest = target.dyn_into().unwrap();
                let error = request.error().unwrap();
                reject.call1(&JsValue::undefined(), &error).unwrap();
            }) as Box<dyn Fn(web_sys::Event)>);

            request.set_onsuccess(Some(success_closure.as_ref().unchecked_ref()));
            request.set_onerror(Some(error_closure.as_ref().unchecked_ref()));

            success_closure.forget();
            error_closure.forget();
        });

        let result = JsFuture::from(promise).await
            .map_err(|e| WebError::web_api_js("Metadata retrieval", e))?;

        if result.is_undefined() {
            return Ok(None);
        }

        let metadata_json = result.as_string().ok_or_else(|| WebError::WebApi {
            api: "Metadata format".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid metadata format")),
        })?;

        let metadata: ModelMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| WebError::WebApi {
                message: format!("Failed to deserialize metadata: {}", e),
            })?;

        Ok(Some(metadata))
    }

    /// List all stored models
    pub async fn list_models(&self) -> WebResult<Vec<ModelMetadata>> {
        let db = self.db.as_ref().ok_or_else(|| WebError::WebApi {
            api: "Database".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::NotConnected, "Database not initialized")),
        })?;

        let transaction = db
            .transaction_with_str(Self::METADATA_STORE)
            .map_err(|e| WebError::web_api_js("Create transaction", e))?;

        let store = transaction
            .object_store(Self::METADATA_STORE)
            .map_err(|e| WebError::web_api_js("Get object store", e))?;

        let request = store
            .get_all()
            .map_err(|e| WebError::web_api_js("Get all metadata", e))?;

        let promise = Promise::new(&mut |resolve, reject| {
            let success_closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
                let target = event.target().unwrap();
                let request: IdbRequest = target.dyn_into().unwrap();
                let result = request.result().unwrap();
                resolve.call1(&JsValue::undefined(), &result).unwrap();
            }) as Box<dyn Fn(web_sys::Event)>);

            let error_closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
                let target = event.target().unwrap();
                let request: IdbRequest = target.dyn_into().unwrap();
                let error = request.error().unwrap();
                reject.call1(&JsValue::undefined(), &error).unwrap();
            }) as Box<dyn Fn(web_sys::Event)>);

            request.set_onsuccess(Some(success_closure.as_ref().unchecked_ref()));
            request.set_onerror(Some(error_closure.as_ref().unchecked_ref()));

            success_closure.forget();
            error_closure.forget();
        });

        let result = JsFuture::from(promise).await
            .map_err(|e| WebError::web_api_js("Model listing", e))?;

        let js_array: Array = result.dyn_into().unwrap();
        let mut models = Vec::new();

        for i in 0..js_array.length() {
            let item = js_array.get(i);
            if let Some(metadata_json) = item.as_string() {
                if let Ok(metadata) = serde_json::from_str::<ModelMetadata>(&metadata_json) {
                    models.push(metadata);
                }
            }
        }

        Ok(models)
    }

    /// Delete a model and its metadata
    pub async fn delete_model(&self, file_id: &FileId) -> WebResult<bool> {
        let db = self.db.as_ref().ok_or_else(|| WebError::WebApi {
            api: "Database".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::NotConnected, "Database not initialized")),
        })?;

        let file_id_str = file_id.to_string();

        // Delete from models store
        let models_transaction = db
            .transaction_with_str_and_mode(Self::MODELS_STORE, IdbTransactionMode::Readwrite)
            .map_err(|e| WebError::web_api_js("Create models transaction", e))?;

        let models_store = models_transaction
            .object_store(Self::MODELS_STORE)
            .map_err(|e| WebError::web_api_js("Get models store", e))?;

        models_store
            .delete(&JsValue::from_str(&file_id_str))
            .map_err(|e| WebError::web_api_js("Delete model", e))?;

        // Delete from metadata store
        let metadata_transaction = db
            .transaction_with_str_and_mode(Self::METADATA_STORE, IdbTransactionMode::Readwrite)
            .map_err(|e| WebError::web_api_js("Create metadata transaction", e))?;

        let metadata_store = metadata_transaction
            .object_store(Self::METADATA_STORE)
            .map_err(|e| WebError::web_api_js("Get metadata store", e))?;

        metadata_store
            .delete(&JsValue::from_str(&file_id_str))
            .map_err(|e| WebError::web_api_js("Delete metadata", e))?;

        // Delete from downloads store
        let downloads_transaction = db
            .transaction_with_str_and_mode(Self::DOWNLOADS_STORE, IdbTransactionMode::Readwrite)
            .map_err(|e| WebError::web_api_js("Create downloads transaction", e))?;

        let downloads_store = downloads_transaction
            .object_store(Self::DOWNLOADS_STORE)
            .map_err(|e| WebError::web_api_js("Get downloads store", e))?;

        downloads_store
            .delete(&JsValue::from_str(&file_id_str))
            .map_err(|e| WebError::web_api_js("Delete progress", e))?;

        Ok(true)
    }

    /// Check storage quota usage
    pub async fn get_storage_quota(&self) -> WebResult<(u64, u64)> {
        let navigator = web_sys::window()
            .ok_or_else(|| WebError::WebApi {
                api: "Window".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::NotFound, "Window not available")),
            })?
            .navigator();

        if let Ok(storage_manager) = navigator.storage() {
            let estimate_promise = storage_manager.estimate();
            let estimate_result = JsFuture::from(estimate_promise).await
                .map_err(|e| WebError::web_api_js("Storage estimate", e))?;

            let estimate_obj = js_sys::Object::from(estimate_result);

            let usage = js_sys::Reflect::get(&estimate_obj, &JsValue::from_str("usage"))
                .unwrap_or(JsValue::from(0))
                .as_f64()
                .unwrap_or(0.0) as u64;

            let quota = js_sys::Reflect::get(&estimate_obj, &JsValue::from_str("quota"))
                .unwrap_or(JsValue::from(0))
                .as_f64()
                .unwrap_or(0.0) as u64;

            Ok((usage, quota))
        } else {
            // Fallback for older browsers
            Ok((0, u64::MAX))
        }
    }

    /// Request persistent storage
    pub async fn request_persistent_storage(&self) -> WebResult<bool> {
        let navigator = web_sys::window()
            .ok_or_else(|| WebError::WebApi {
                api: "Window".to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::NotFound, "Window not available")),
            })?
            .navigator();

        if let Ok(storage_manager) = navigator.storage() {
            let persist_promise = storage_manager.persist();
            let persist_result = JsFuture::from(persist_promise).await
                .map_err(|e| WebError::web_api_js("Persist request", e))?;

            Ok(persist_result.as_bool().unwrap_or(false))
        } else {
            Ok(false)
        }
    }
}