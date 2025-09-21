//! Web storage implementation using browser APIs
//! Uses `IndexedDB` for persistent storage

use crate::error::{format_js_error, StorageError, WebError, WebResult};
use js_sys::{Array, Promise, Uint8Array};
use purr_common::platform::FileId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{oneshot, RwLock};
use uuid::Uuid;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys::{IdbDatabase, IdbOpenDbRequest, IdbVersionChangeEvent};

/// File metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub id: String,
    pub original_name: String,
    pub mime_type: String,
    pub size: usize,
    pub created_at: f64,
    pub last_accessed: f64,
}

/// Commands for `IndexedDB` operations
#[derive(Debug)]
enum IndexedDBCommand {
    Initialize {
        db_name: String,
        response: oneshot::Sender<WebResult<()>>,
    },
    StoreFile {
        file_id: String,
        data: Vec<u8>,
        metadata: FileMetadata,
        response: oneshot::Sender<WebResult<()>>,
    },
    GetFile {
        file_id: String,
        response: oneshot::Sender<WebResult<Option<Vec<u8>>>>,
    },
    DeleteFile {
        file_id: String,
        response: oneshot::Sender<WebResult<()>>,
    },
    LoadMetadata {
        response: oneshot::Sender<WebResult<HashMap<String, FileMetadata>>>,
    },
}

/// Web storage using `IndexedDB`
#[derive(Debug)]
pub struct WebStorage {
    db_name: String,
    metadata_cache: Arc<RwLock<HashMap<String, FileMetadata>>>,
    command_sender: tokio::sync::mpsc::UnboundedSender<IndexedDBCommand>,
}

impl WebStorage {
    /// Create `WebStorage` instance
    #[must_use]
    pub fn new() -> Self {
        let (command_sender, mut command_receiver) = tokio::sync::mpsc::unbounded_channel();

        // Spawn the IndexedDB command processor
        spawn_local(async move {
            let mut db: Option<IdbDatabase> = None;

            while let Some(command) = command_receiver.recv().await {
                match command {
                    IndexedDBCommand::Initialize { db_name, response } => {
                        let result = Self::initialize_database(&db_name).await;
                        match result {
                            Ok(database) => {
                                db = Some(database);
                                let _ = response.send(Ok(()));
                            }
                            Err(e) => {
                                let _ = response.send(Err(e));
                            }
                        }
                    }
                    IndexedDBCommand::StoreFile {
                        file_id,
                        data,
                        metadata,
                        response,
                    } => {
                        let result = if let Some(ref database) = db {
                            Self::store_file_internal(database, &file_id, &data, &metadata).await
                        } else {
                            Err(WebError::from(StorageError::DatabaseConnectionFailed))
                        };
                        let _ = response.send(result);
                    }
                    IndexedDBCommand::GetFile { file_id, response } => {
                        let result = if let Some(ref database) = db {
                            Self::get_file_internal(database, &file_id).await
                        } else {
                            Err(WebError::from(StorageError::DatabaseConnectionFailed))
                        };
                        let _ = response.send(result);
                    }
                    IndexedDBCommand::DeleteFile { file_id, response } => {
                        let result = if let Some(ref database) = db {
                            Self::delete_file_internal(database, &file_id).await
                        } else {
                            Err(WebError::from(StorageError::DatabaseConnectionFailed))
                        };
                        let _ = response.send(result);
                    }
                    IndexedDBCommand::LoadMetadata { response } => {
                        let result = if let Some(ref database) = db {
                            Self::load_metadata_internal(database).await
                        } else {
                            Err(WebError::from(StorageError::DatabaseConnectionFailed))
                        };
                        let _ = response.send(result);
                    }
                }
            }
        });

        Self {
            db_name: "PurrWebStorage".to_string(),
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            command_sender,
        }
    }

    /// Store file in `IndexedDB`
    pub async fn store_file(
        &self,
        data: bytes::Bytes,
        filename: &str,
        mime_type: &str,
    ) -> WebResult<FileId> {
        let file_id = FileId::new(Uuid::new_v4().to_string());
        let timestamp = js_sys::Date::now();

        let metadata = FileMetadata {
            id: file_id.to_string(),
            original_name: filename.to_string(),
            mime_type: mime_type.to_string(),
            size: data.len(),
            created_at: timestamp,
            last_accessed: timestamp,
        };

        // Store in metadata cache
        {
            let mut cache = self.metadata_cache.write().await;
            cache.insert(file_id.to_string(), metadata.clone());
        }

        // Store file data in IndexedDB using command system
        let (response_tx, response_rx) = oneshot::channel();
        let command = IndexedDBCommand::StoreFile {
            file_id: file_id.to_string(),
            data: data.to_vec(),
            metadata,
            response: response_tx,
        };

        self.command_sender
            .send(command)
            .map_err(|_| WebError::from(StorageError::ObjectStoreAccessFailed))?;

        response_rx
            .await
            .map_err(|e| WebError::from(StorageError::GetOperationFailed))??;

        tracing::info!("Stored file: {} ({} bytes)", filename, data.len());
        Ok(file_id)
    }

    /// Get file from `IndexedDB`
    pub async fn get_file(&self, file_id: &FileId) -> WebResult<Option<bytes::Bytes>> {
        // Update last accessed
        {
            let mut cache = self.metadata_cache.write().await;
            if let Some(metadata) = cache.get_mut(&file_id.to_string()) {
                metadata.last_accessed = js_sys::Date::now();
            }
        }

        // Get file data from IndexedDB using command system
        let (response_tx, response_rx) = oneshot::channel();
        let command = IndexedDBCommand::GetFile {
            file_id: file_id.to_string(),
            response: response_tx,
        };

        self.command_sender
            .send(command)
            .map_err(|_| WebError::from(StorageError::ObjectStoreAccessFailed))?;

        let file_data = response_rx
            .await
            .map_err(|e| WebError::from(StorageError::GetOperationFailed))??;

        tracing::info!("Retrieved file: {}", file_id);
        Ok(file_data.map(bytes::Bytes::from))
    }

    /// Delete file from `IndexedDB`
    pub async fn delete_file(&self, file_id: &FileId) -> WebResult<()> {
        // Remove from metadata cache
        {
            let mut cache = self.metadata_cache.write().await;
            cache.remove(&file_id.to_string());
        }

        // Delete file data from IndexedDB using command system
        let (response_tx, response_rx) = oneshot::channel();
        let command = IndexedDBCommand::DeleteFile {
            file_id: file_id.to_string(),
            response: response_tx,
        };

        self.command_sender
            .send(command)
            .map_err(|_| WebError::from(StorageError::ObjectStoreAccessFailed))?;

        response_rx
            .await
            .map_err(|e| WebError::from(StorageError::GetOperationFailed))??;

        tracing::info!("Deleted file: {}", file_id);
        Ok(())
    }

    /// Get file metadata
    pub async fn get_metadata(&self, file_id: &FileId) -> WebResult<Option<FileMetadata>> {
        let cache = self.metadata_cache.read().await;
        Ok(cache.get(&file_id.to_string()).cloned())
    }

    /// List all files
    pub async fn list_files(&self) -> WebResult<Vec<FileMetadata>> {
        let cache = self.metadata_cache.read().await;
        Ok(cache.values().cloned().collect())
    }

    /// Initialize the `IndexedDB` database
    pub async fn initialize(&self) -> WebResult<()> {
        let (response_tx, response_rx) = oneshot::channel();
        let command = IndexedDBCommand::Initialize {
            db_name: self.db_name.clone(),
            response: response_tx,
        };

        self.command_sender
            .send(command)
            .map_err(|_| WebError::from(StorageError::ObjectStoreAccessFailed))?;

        response_rx
            .await
            .map_err(|e| WebError::from(StorageError::GetOperationFailed))??;

        // Load existing metadata from IndexedDB
        self.load_metadata().await?;

        tracing::info!("Initialized IndexedDB storage");
        Ok(())
    }

    /// Load metadata from `IndexedDB`
    async fn load_metadata(&self) -> WebResult<()> {
        let (response_tx, response_rx) = oneshot::channel();
        let command = IndexedDBCommand::LoadMetadata {
            response: response_tx,
        };

        self.command_sender
            .send(command)
            .map_err(|_| WebError::from(StorageError::ObjectStoreAccessFailed))?;

        let metadata_map = response_rx
            .await
            .map_err(|e| WebError::from(StorageError::GetOperationFailed))??;

        // Update cache with loaded metadata
        {
            let mut cache = self.metadata_cache.write().await;
            *cache = metadata_map;
        }

        tracing::info!("Loaded metadata from IndexedDB");
        Ok(())
    }

    /// Initialize database and create object stores
    async fn initialize_database(db_name: &str) -> WebResult<IdbDatabase> {
        let window = web_sys::window().ok_or(StorageError::IndexedDbUnavailable)?;

        let idb = window
            .indexed_db()
            .map_err(|e| StorageError::IndexedDbUnavailableJs { js_error: format_js_error(e) })?
            .ok_or(StorageError::IndexedDbUnavailable)?;

        let open_request = idb
            .open_with_u32(db_name, 1)
            .map_err(|e| StorageError::DatabaseOpenFailedJs { js_error: format_js_error(e) })?;

        // Set up upgrade handler
        let upgrade_closure = Closure::wrap(Box::new(move |event: IdbVersionChangeEvent| {
            let target = event.target().unwrap();
            let request: IdbOpenDbRequest = target.dyn_into().unwrap();
            let db: IdbDatabase = request.result().unwrap().dyn_into().unwrap();

            // Create object stores
            if !db.object_store_names().contains("files") {
                let _ = db.create_object_store("files");
            }
            if !db.object_store_names().contains("metadata") {
                let _ = db.create_object_store("metadata");
            }
        }) as Box<dyn FnMut(_)>);

        open_request.set_onupgradeneeded(Some(upgrade_closure.as_ref().unchecked_ref()));
        upgrade_closure.forget();

        // Wait for database to open
        let promise = Promise::from(JsValue::from(open_request));
        let db_result = JsFuture::from(promise)
            .await
            .map_err(|e| StorageError::DatabaseConnectionFailedJs { js_error: format_js_error(e) })?;

        let db: IdbDatabase = db_result
            .dyn_into()
            .map_err(|e| StorageError::DatabaseCastFailedJs { js_error: format_js_error(e) })?;

        Ok(db)
    }

    /// Store file data and metadata in `IndexedDB`
    async fn store_file_internal(
        db: &IdbDatabase,
        file_id: &str,
        data: &[u8],
        metadata: &FileMetadata,
    ) -> WebResult<()> {
        // Store file data
        let transaction = db
            .transaction_with_str_and_mode("files", web_sys::IdbTransactionMode::Readwrite)
            .map_err(|e| StorageError::TransactionCreationFailedJs { js_error: format_js_error(e) })?;

        let files_store = transaction
            .object_store("files")
            .map_err(|e| StorageError::ObjectStoreAccessFailedJs { js_error: format_js_error(e) })?;

        // Convert bytes to Uint8Array
        let array = Uint8Array::new_with_length(data.len() as u32);
        array.copy_from(data);

        let file_request = files_store
            .put_with_key(&array.into(), &JsValue::from_str(file_id))
            .map_err(|e| WebError::from(StorageError::FileStoreFailedJs { js_error: format_js_error(e) }))?;

        let promise = Promise::from(JsValue::from(file_request));
        JsFuture::from(promise)
            .await
            .map_err(|e| WebError::from(StorageError::FileStorageOperationFailedJs { js_error: format_js_error(e) }))?;

        // Store metadata
        let metadata_transaction = db
            .transaction_with_str_and_mode("metadata", web_sys::IdbTransactionMode::Readwrite)
            .map_err(|e| WebError::from(StorageError::MetadataTransactionFailed))?;

        let metadata_store = metadata_transaction
            .object_store("metadata")
            .map_err(|e| WebError::from(StorageError::MetadataStoreAccessFailed))?;

        let serialized = serde_wasm_bindgen::to_value(metadata)
            .map_err(|e| WebError::from(StorageError::MetadataSerializationFailed))?;

        let metadata_request = metadata_store
            .put_with_key(&serialized, &JsValue::from_str(file_id))
            .map_err(|e| WebError::from(StorageError::MetadataStoreFailedJs { js_error: format_js_error(e) }))?;

        let promise = Promise::from(JsValue::from(metadata_request));
        JsFuture::from(promise)
            .await
            .map_err(|e| WebError::from(StorageError::MetadataStorageOperationFailedJs { js_error: format_js_error(e) }))?;

        Ok(())
    }

    /// Get file data from `IndexedDB`
    async fn get_file_internal(db: &IdbDatabase, file_id: &str) -> WebResult<Option<Vec<u8>>> {
        let transaction = db
            .transaction_with_str("files")
            .map_err(|_| WebError::from(StorageError::TransactionCreationFailed))?;

        let object_store = transaction
            .object_store("files")
            .map_err(|_| WebError::from(StorageError::ObjectStoreAccessFailed))?;

        let request = object_store
            .get(&JsValue::from_str(file_id))
            .map_err(|e| WebError::from(StorageError::GetRequestFailedJs { js_error: format_js_error(e) }))?;

        let promise = Promise::from(JsValue::from(request));
        let result = JsFuture::from(promise)
            .await
            .map_err(|_| WebError::from(StorageError::GetOperationFailed))?;

        if result.is_undefined() {
            return Ok(None);
        }

        // Convert Uint8Array back to Vec<u8>
        let array: Uint8Array = result
            .dyn_into()
            .map_err(|_| WebError::from(StorageError::DataFormatInvalid))?;

        let mut data = vec![0u8; array.length() as usize];
        array.copy_to(&mut data);

        Ok(Some(data))
    }

    /// Delete file and metadata from `IndexedDB`
    async fn delete_file_internal(db: &IdbDatabase, file_id: &str) -> WebResult<()> {
        // Delete file data
        let file_transaction = db
            .transaction_with_str_and_mode("files", web_sys::IdbTransactionMode::Readwrite)
            .map_err(|_| WebError::from(StorageError::TransactionCreationFailed))?;

        let files_store = file_transaction
            .object_store("files")
            .map_err(|_| WebError::from(StorageError::ObjectStoreAccessFailed))?;

        let file_request = files_store
            .delete(&JsValue::from_str(file_id))
            .map_err(|e| WebError::from(StorageError::DeleteOperationFailedJs { js_error: format_js_error(e) }))?;

        let promise = Promise::from(JsValue::from(file_request));
        JsFuture::from(promise)
            .await
            .map_err(|e| WebError::from(StorageError::FileDeletionFailedJs { js_error: format_js_error(e) }))?;

        // Delete metadata
        let metadata_transaction = db
            .transaction_with_str_and_mode("metadata", web_sys::IdbTransactionMode::Readwrite)
            .map_err(|e| WebError::from(StorageError::MetadataTransactionFailed))?;

        let metadata_store = metadata_transaction
            .object_store("metadata")
            .map_err(|e| WebError::from(StorageError::MetadataStoreAccessFailed))?;

        let metadata_request = metadata_store
            .delete(&JsValue::from_str(file_id))
            .map_err(|e| WebError::from(StorageError::MetadataDeleteFailed))?;

        let promise = Promise::from(JsValue::from(metadata_request));
        JsFuture::from(promise)
            .await
            .map_err(|e| WebError::from(StorageError::MetadataDeletionFailedJs { js_error: format_js_error(e) }))?;

        Ok(())
    }

    /// Load all metadata from `IndexedDB`
    async fn load_metadata_internal(db: &IdbDatabase) -> WebResult<HashMap<String, FileMetadata>> {
        let transaction = db
            .transaction_with_str("metadata")
            .map_err(|_| WebError::from(StorageError::TransactionCreationFailed))?;

        let object_store = transaction
            .object_store("metadata")
            .map_err(|_| WebError::from(StorageError::ObjectStoreAccessFailed))?;

        let request = object_store
            .get_all()
            .map_err(|e| WebError::from(StorageError::GetAllRequestFailedJs { js_error: format_js_error(e) }))?;

        let promise = Promise::from(JsValue::from(request));
        let result = JsFuture::from(promise)
            .await
            .map_err(|e| WebError::from(StorageError::GetAllOperationFailedJs { js_error: format_js_error(e) }))?;

        let array: Array = result
            .dyn_into()
            .map_err(|_| WebError::from(StorageError::ArrayFormatInvalid))?;

        let mut metadata_map = HashMap::new();

        for i in 0..array.length() {
            let item = array.get(i);
            if let Ok(metadata) = serde_wasm_bindgen::from_value::<FileMetadata>(item) {
                metadata_map.insert(metadata.id.clone(), metadata);
            }
        }

        Ok(metadata_map)
    }
}

impl Default for WebStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_files: usize,
    pub total_size: usize,
    pub indexeddb_available: bool,
}
