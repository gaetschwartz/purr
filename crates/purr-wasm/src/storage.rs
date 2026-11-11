//! Web storage implementation using browser APIs
//! Uses `IndexedDB` for persistent storage

use crate::error::{StorageError, WebError, WebResult, format_js_error};
use crate::idb::{self, FILES, METADATA};
use js_sys::{Array, Uint8Array};
use purr_common::platform::FileId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, oneshot};
use uuid::Uuid;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::{IdbDatabase, IdbObjectStore, IdbTransactionMode};

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

fn reply<T>(response: oneshot::Sender<WebResult<T>>, result: WebResult<T>) {
    if response.send(result).is_err() {
        tracing::debug!("A storage request finished after its caller left");
    }
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
                                reply(response, Ok(()));
                            }
                            Err(e) => {
                                reply(response, Err(e));
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
                            Err(WebError::from(StorageError::DatabaseConnectionFailed {
                                js_error: "Database not available".to_owned(),
                            }))
                        };
                        reply(response, result);
                    }
                    IndexedDBCommand::GetFile { file_id, response } => {
                        let result = if let Some(ref database) = db {
                            Self::get_file_internal(database, &file_id).await
                        } else {
                            Err(WebError::from(StorageError::DatabaseConnectionFailed {
                                js_error: "Database not available".to_owned(),
                            }))
                        };
                        reply(response, result);
                    }
                    IndexedDBCommand::DeleteFile { file_id, response } => {
                        let result = if let Some(ref database) = db {
                            Self::delete_file_internal(database, &file_id).await
                        } else {
                            Err(WebError::from(StorageError::DatabaseConnectionFailed {
                                js_error: "Database not available".to_owned(),
                            }))
                        };
                        reply(response, result);
                    }
                    IndexedDBCommand::LoadMetadata { response } => {
                        let result = if let Some(ref database) = db {
                            Self::load_metadata_internal(database).await
                        } else {
                            Err(WebError::from(StorageError::DatabaseConnectionFailed {
                                js_error: "Database not available".to_owned(),
                            }))
                        };
                        reply(response, result);
                    }
                }
            }
        });

        Self {
            db_name: "PurrWebStorage".to_owned(),
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
            original_name: filename.to_owned(),
            mime_type: mime_type.to_owned(),
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

        self.command_sender.send(command).map_err(|_| {
            WebError::from(StorageError::ObjectStoreAccessFailed {
                js_error: "Command send failed".to_owned(),
            })
        })?;

        response_rx.await.map_err(|e| {
            WebError::from(StorageError::GetOperationFailed {
                js_error: format!("{e:?}"),
            })
        })??;

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

        self.command_sender.send(command).map_err(|_| {
            WebError::from(StorageError::ObjectStoreAccessFailed {
                js_error: "Command send failed".to_owned(),
            })
        })?;

        let file_data = response_rx.await.map_err(|e| {
            WebError::from(StorageError::GetOperationFailed {
                js_error: format!("{e:?}"),
            })
        })??;

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

        self.command_sender.send(command).map_err(|_| {
            WebError::from(StorageError::ObjectStoreAccessFailed {
                js_error: "Command send failed".to_owned(),
            })
        })?;

        response_rx.await.map_err(|e| {
            WebError::from(StorageError::GetOperationFailed {
                js_error: format!("{e:?}"),
            })
        })??;

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

    /// Deletes every stored file that was created more than `max_age_ms` ago.
    pub async fn purge_older_than(&self, max_age_ms: f64) -> WebResult<()> {
        let now = js_sys::Date::now();
        for file in self.list_files().await? {
            if now - file.created_at > max_age_ms {
                self.delete_file(&FileId::from(file.id.as_str())).await?;
                tracing::info!("Removed stale upload {}", file.original_name);
            }
        }
        Ok(())
    }

    /// Initialize the `IndexedDB` database
    pub async fn initialize(&self) -> WebResult<()> {
        let (response_tx, response_rx) = oneshot::channel();
        let command = IndexedDBCommand::Initialize {
            db_name: self.db_name.clone(),
            response: response_tx,
        };

        self.command_sender.send(command).map_err(|_| {
            WebError::from(StorageError::ObjectStoreAccessFailed {
                js_error: "Command send failed".to_owned(),
            })
        })?;

        response_rx.await.map_err(|e| {
            WebError::from(StorageError::GetOperationFailed {
                js_error: format!("{e:?}"),
            })
        })??;

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

        self.command_sender.send(command).map_err(|_| {
            WebError::from(StorageError::ObjectStoreAccessFailed {
                js_error: "Command send failed".to_owned(),
            })
        })?;

        let metadata_map = response_rx.await.map_err(|e| {
            WebError::from(StorageError::GetOperationFailed {
                js_error: format!("{e:?}"),
            })
        })??;

        // Update cache with loaded metadata
        {
            let mut cache = self.metadata_cache.write().await;
            *cache = metadata_map;
        }

        tracing::info!("Loaded metadata from IndexedDB");
        Ok(())
    }

    /// Opens the shared database, creating its object stores on first use or upgrade
    async fn initialize_database(_db_name: &str) -> WebResult<IdbDatabase> {
        crate::idb::connect().await
    }

    /// Store file data and metadata in `IndexedDB`
    async fn store_file_internal(
        db: &IdbDatabase,
        file_id: &str,
        data: &[u8],
        metadata: &FileMetadata,
    ) -> WebResult<()> {
        let files_store = Self::store(db, FILES, IdbTransactionMode::Readwrite)?;
        let array = Uint8Array::from(data);
        let file_request = files_store
            .put_with_key(&array.into(), &JsValue::from_str(file_id))
            .map_err(|e| StorageError::FileStoreFailed {
                js_error: format_js_error(e),
            })?;
        idb::settle(&file_request, "store upload").await?;

        let metadata_store = Self::store(db, METADATA, IdbTransactionMode::Readwrite)?;
        let serialized = serde_wasm_bindgen::to_value(metadata)
            .map_err(|_e| StorageError::MetadataSerializationFailed)?;
        let metadata_request = metadata_store
            .put_with_key(&serialized, &JsValue::from_str(file_id))
            .map_err(|e| StorageError::MetadataStoreFailed {
                js_error: format_js_error(e),
            })?;
        idb::settle(&metadata_request, "store upload details").await?;
        Ok(())
    }

    fn store(db: &IdbDatabase, name: &str, mode: IdbTransactionMode) -> WebResult<IdbObjectStore> {
        let transaction = db.transaction_with_str_and_mode(name, mode).map_err(|e| {
            StorageError::TransactionCreationFailed {
                js_error: format_js_error(e),
            }
        })?;
        Ok(transaction
            .object_store(name)
            .map_err(|e| StorageError::ObjectStoreAccessFailed {
                js_error: format_js_error(e),
            })?)
    }

    /// Get file data from `IndexedDB`
    async fn get_file_internal(db: &IdbDatabase, file_id: &str) -> WebResult<Option<Vec<u8>>> {
        let object_store = Self::store(db, FILES, IdbTransactionMode::Readonly)?;
        let request = object_store.get(&JsValue::from_str(file_id)).map_err(|e| {
            StorageError::GetRequestFailed {
                js_error: format_js_error(e),
            }
        })?;
        let result = idb::settle(&request, "read upload").await?;
        if result.is_undefined() {
            return Ok(None);
        }
        let array: Uint8Array = result
            .dyn_into()
            .map_err(|_| StorageError::DataFormatInvalid)?;
        Ok(Some(array.to_vec()))
    }

    /// Delete file and metadata from `IndexedDB`
    async fn delete_file_internal(db: &IdbDatabase, file_id: &str) -> WebResult<()> {
        let files_store = Self::store(db, FILES, IdbTransactionMode::Readwrite)?;
        let file_request = files_store
            .delete(&JsValue::from_str(file_id))
            .map_err(|e| StorageError::DeleteOperationFailed {
                js_error: format_js_error(e),
            })?;
        idb::settle(&file_request, "delete upload").await?;

        let metadata_store = Self::store(db, METADATA, IdbTransactionMode::Readwrite)?;
        let metadata_request = metadata_store
            .delete(&JsValue::from_str(file_id))
            .map_err(|e| StorageError::MetadataDeleteFailed {
                js_error: format_js_error(e),
            })?;
        idb::settle(&metadata_request, "delete upload details").await?;
        Ok(())
    }

    /// Load all metadata from `IndexedDB`
    async fn load_metadata_internal(db: &IdbDatabase) -> WebResult<HashMap<String, FileMetadata>> {
        let object_store = Self::store(db, METADATA, IdbTransactionMode::Readonly)?;
        let request = object_store
            .get_all()
            .map_err(|e| StorageError::GetAllRequestFailed {
                js_error: format_js_error(e),
            })?;
        let result = idb::settle(&request, "read upload details").await?;
        let array: Array = result
            .dyn_into()
            .map_err(|_| StorageError::ArrayFormatInvalid)?;

        let mut metadata_map = HashMap::new();
        for item in array.iter() {
            match serde_wasm_bindgen::from_value::<FileMetadata>(item) {
                Ok(metadata) => {
                    metadata_map.insert(metadata.id.clone(), metadata);
                }
                Err(error) => tracing::warn!("Skipped unreadable upload details: {error}"),
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
