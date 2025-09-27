/// Default platform implementation for desktop/native builds
/// This implementation uses direct file system access and native whisper transcription
use super::{Platform, PlatformError, TranscriptionRequest, TranscriptionStatus};
use bytes::Bytes;
use futures::{channel::mpsc, SinkExt, Stream, StreamExt};
use purr_common::platform::{
    DeviceInfo, DeviceType, FileId, FileSource, ModelInfo, ModelMetadata, ModelOperationProgress,
};
use purr_core::dev;
use purr_core::model::{ModelManager, WhisperModel};
use std::{
    path::{Path, PathBuf},
    pin::Pin,
    str::FromStr,
    sync::Arc,
};
use tokio::sync::Mutex;
use tracing::{error, info, warn};

pub(crate) type PlatformImpl = DesktopPlatformImpl;

pub(super) struct DesktopPlatformImpl {
    temp_dir: PathBuf,
    model_manager: Arc<Mutex<ModelManager>>,
}

impl DesktopPlatformImpl {
    pub fn new() -> Self {
        let temp_dir = std::env::temp_dir().join("purr-temp");
        // Ensure temp directory exists
        let _ = std::fs::create_dir_all(&temp_dir);

        // Initialize model manager (will create XDG-compliant directories)
        let model_manager = ModelManager::new()
            .map(|mm| Arc::new(Mutex::new(mm)))
            .unwrap_or_else(|err| {
                warn!("Failed to initialize model manager: {}", err);
                // Create a fallback model manager with temp directory
                Arc::new(Mutex::new(ModelManager::default()))
            });

        Self {
            temp_dir,
            model_manager,
        }
    }
}

#[async_trait::async_trait]
impl Platform for DesktopPlatformImpl {
    async fn new() -> Result<Self, PlatformError> {
        Ok(Self::new())
    }

    async fn process_file(
        &self,
        file_data: Bytes,
        file_path: &Path,
    ) -> Result<FileId, PlatformError> {
        use std::fs::File;
        use std::io::Write;
        use uuid::Uuid;

        // Generate unique file ID
        let file_id =
            Uuid::new_v3(&Uuid::NAMESPACE_URL, file_path.to_string_lossy().as_bytes()).into();
        let file_path = self.temp_dir.join(&file_id);

        info!(
            "Processing file {} with ID: {}",
            file_path.display(),
            file_id
        );

        // Write file data to temp location
        let mut file = File::create(&file_path).map_err(PlatformError::io)?;
        file.write_all(&file_data).map_err(PlatformError::io)?;
        file.flush().map_err(PlatformError::io)?;

        info!("File saved to: {:?}", file_path);
        Ok(file_id)
    }

    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<TranscriptionStatus, PlatformError>> + Send>>,
        PlatformError,
    > {
        use futures::channel::mpsc;
        use futures::SinkExt;
        use purr_core::{transcribe_file_stream, TranscriptionConfig};

        // Create a channel for status updates
        let (mut tx, rx) = mpsc::channel(10);
        let tmp_dir = self.temp_dir.clone();

        tokio::spawn(async move {
            let start_time = std::time::Instant::now();

            // Send starting status
            let _ = tx.send(Ok(TranscriptionStatus::Starting)).await;
            info!("Starting transcription for {}", request.file);

            // Create transcription configuration
            let mut config = TranscriptionConfig::new().with_translate(request.translate);
            if let Some(language) = request.language {
                config = config.with_language(language);
            }

            // Send processing audio status
            let _ = tx.send(Ok(TranscriptionStatus::ProcessingAudio)).await;

            let audio_path = match &request.file {
                FileSource::Bytes(_) => panic!("Bytes source not supported in desktop platform"),
                FileSource::Path(path) => path,
                FileSource::Uploaded(file_id) => &tmp_dir.join(file_id),
            };
            // Start transcription
            match transcribe_file_stream(audio_path, config).await {
                Ok(mut stream) => {
                    let mut word_count = 0;
                    let mut audio_duration = 0.0f32;

                    // Process streaming chunks
                    while let Some(chunk_result) = stream.next().await {
                        match chunk_result {
                            Ok(chunk) => {
                                word_count += chunk.text.split_whitespace().count();

                                // Send progress update
                                let _ = tx
                                    .send(Ok(TranscriptionStatus::InProgress {
                                        chunk_index: chunk.chunk_index,
                                        text: chunk.text,
                                        start_time: chunk.start,
                                        end_time: chunk.end,
                                    }))
                                    .await;

                                // Check for final stats
                                if let Some(stats) = chunk.final_stats {
                                    audio_duration = stats.audio_duration;
                                    word_count = stats.word_count;
                                    break;
                                }
                            }
                            Err(e) => {
                                error!("Transcription error: {}", e);
                                let _ = tx.send(Err(PlatformError::transcription(e))).await;
                                return;
                            }
                        }
                    }

                    // Send completion status
                    let processing_time = start_time.elapsed().as_secs_f64();
                    let _ = tx
                        .send(Ok(TranscriptionStatus::Completed {
                            processing_time,
                            audio_duration,
                            word_count,
                        }))
                        .await;

                    info!("Transcription completed in {:.2}s", processing_time);
                }
                Err(e) => {
                    error!("Failed to start transcription: {}", e);
                    let _ = tx.send(Err(PlatformError::transcription(e))).await;
                }
            }
        });

        Ok(Box::pin(rx))
    }

    async fn cleanup(&self, file_id: &str) -> Result<(), PlatformError> {
        let file_path = self.temp_dir.join(file_id);
        if file_path.exists() {
            std::fs::remove_file(&file_path).map_err(PlatformError::io)?;
            info!("Cleaned up temporary file: {}", file_id);
        }
        Ok(())
    }

    // ========================================
    // Model Management Implementation
    // ========================================

    async fn list_installed_models(&self) -> Result<Vec<ModelInfo>, PlatformError> {
        let manager = self.model_manager.lock().await;
        let downloaded_models = manager
            .list_downloaded_models()
            .await
            .map_err(PlatformError::model_management)?;

        let mut model_infos = Vec::new();
        for model in downloaded_models {
            let model_path = manager.get_model_path(model);
            let metadata = self.create_model_metadata(&model);

            let mut model_info = ModelInfo::new(
                model.as_str().to_string(),
                model.as_str().to_string(),
                model.description().to_string(),
                model.size(),
                true,
                metadata,
            );
            model_info.mark_installed(model_path);
            model_infos.push(model_info);
        }

        Ok(model_infos)
    }

    async fn list_available_models(&self) -> Result<Vec<ModelInfo>, PlatformError> {
        let mut model_infos = Vec::new();
        let manager = self.model_manager.lock().await;

        for &model in WhisperModel::all_models() {
            let is_installed = manager.is_model_downloaded(model).await;
            let model_path = if is_installed {
                Some(manager.get_model_path(model))
            } else {
                None
            };

            let metadata = self.create_model_metadata(&model);

            let mut model_info = ModelInfo::new(
                model.as_str().to_string(),
                model.as_str().to_string(),
                model.description().to_string(),
                model.size(),
                is_installed,
                metadata,
            );

            if let Some(path) = model_path {
                model_info.mark_installed(path);
            }

            model_infos.push(model_info);
        }

        Ok(model_infos)
    }

    async fn fetch_model(
        &self,
        model_id: &str,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ModelOperationProgress, PlatformError>> + Send>>,
        PlatformError,
    > {
        let model = WhisperModel::from_str(model_id)
            .map_err(|_| PlatformError::model_not_found(model_id.to_string()))?;

        let manager = Arc::clone(&self.model_manager);
        let model_id = model_id.to_string();

        let (mut tx, rx) = mpsc::channel(10);

        tokio::spawn(async move {
            let _ = tx
                .send(Ok(ModelOperationProgress::Starting {
                    model_id: model_id.clone(),
                }))
                .await;

            let manager = manager.lock().await;

            // Check if already installed
            if manager.is_model_downloaded(model).await {
                let model_path = manager.get_model_path(model);
                let _ = tx
                    .send(Ok(ModelOperationProgress::Completed {
                        model_id: model_id.clone(),
                        local_path: model_path,
                    }))
                    .await;
                return;
            }

            // Download with progress
            match manager
                .download_model_with_progress(model, |downloaded, total| {
                    let progress = ModelOperationProgress::Downloading {
                        model_id: model_id.clone(),
                        bytes_downloaded: downloaded,
                        total_bytes: total,
                        speed_bps: None, // Could calculate from time intervals
                    };
                    // Send progress (ignore errors as channel might be closed)
                    let _ = tx.try_send(Ok(progress));
                })
                .await
            {
                Ok(model_path) => {
                    let _ = tx
                        .send(Ok(ModelOperationProgress::Completed {
                            model_id: model_id.clone(),
                            local_path: model_path,
                        }))
                        .await;
                }
                Err(e) => {
                    let _ = tx
                        .send(Ok(ModelOperationProgress::Failed {
                            model_id: model_id.clone(),
                            operation: "model_loading".to_string(),
                            error_message: e.to_string(),
                        }))
                        .await;
                }
            }
        });

        Ok(Box::pin(rx))
    }

    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo, PlatformError> {
        let model = WhisperModel::from_str(model_id)
            .map_err(|_| PlatformError::model_not_found(model_id.to_string()))?;

        let manager = self.model_manager.lock().await;
        let is_installed = manager.is_model_downloaded(model).await;
        let model_path = if is_installed {
            Some(manager.get_model_path(model))
        } else {
            None
        };

        let metadata = self.create_model_metadata(&model);

        let mut model_info = ModelInfo::new(
            model.as_str().to_string(),
            model.as_str().to_string(),
            model.description().to_string(),
            model.size(),
            is_installed,
            metadata,
        );

        if let Some(path) = model_path {
            model_info.mark_installed(path);
        }

        Ok(model_info)
    }

    async fn remove_model(&self, model_id: &str) -> Result<(), PlatformError> {
        let model = WhisperModel::from_str(model_id)
            .map_err(|_| PlatformError::model_not_found(model_id.to_string()))?;

        let manager = self.model_manager.lock().await;
        manager
            .delete_model(model)
            .await
            .map_err(PlatformError::model_management)?;

        Ok(())
    }

    async fn is_model_installed(&self, model_id: &str) -> Result<bool, PlatformError> {
        let model = WhisperModel::from_str(model_id)
            .map_err(|_| PlatformError::model_not_found(model_id.to_string()))?;

        let manager = self.model_manager.lock().await;
        Ok(manager.is_model_downloaded(model).await)
    }

    async fn get_model_path(&self, model_id: &str) -> Result<String, PlatformError> {
        let model = WhisperModel::from_str(model_id)
            .map_err(|_| PlatformError::model_not_found(model_id.to_string()))?;

        let manager = self.model_manager.lock().await;
        if !manager.is_model_downloaded(model).await {
            return Err(PlatformError::model_not_found(model_id.to_string()));
        }

        let model_path = manager.get_model_path(model);
        Ok(model_path.to_string_lossy().to_string())
    }

    async fn list_available_devices(&self) -> Result<Vec<DeviceInfo>, PlatformError> {
        // Use the same logic as purr_core::dev::list_devices()
        let devices = dev::list_devices();

        let device_infos = devices
            .into_iter()
            .map(|device| DeviceInfo {
                id: device.id,
                name: device.name,
                description: Some(device.description),
                device_type: match device.tpe {
                    dev::DeviceType::Cpu => DeviceType::Cpu,
                    dev::DeviceType::Gpu => DeviceType::Gpu,
                    dev::DeviceType::Accel => DeviceType::Accel,
                    dev::DeviceType::Unknown => DeviceType::Unknown,
                },
                memory_free: Some(device.vram_free),
                memory_total: Some(device.vram_total),
                capabilities: device.caps.map(|caps| {
                    let mut cap_map = std::collections::HashMap::new();
                    cap_map.insert("async".to_string(), serde_json::Value::Bool(caps.async_));
                    cap_map.insert(
                        "host_buffer".to_string(),
                        serde_json::Value::Bool(caps.host_buffer),
                    );
                    cap_map.insert(
                        "buffer_from_host_ptr".to_string(),
                        serde_json::Value::Bool(caps.buffer_from_host_ptr),
                    );
                    cap_map.insert("events".to_string(), serde_json::Value::Bool(caps.events));
                    cap_map
                }),
            })
            .collect();

        Ok(device_infos)
    }
}

impl DesktopPlatformImpl {
    /// Helper method to create model metadata for `WhisperModel`
    fn create_model_metadata(&self, model: &WhisperModel) -> ModelMetadata {
        ModelMetadata::new(
            "ggml".to_string(),
            "whisper".to_string(),
            if model.as_str().contains(".en") {
                "en-only".to_string()
            } else {
                "multilingual".to_string()
            },
            "v1".to_string(),
        )
        .with_download_url(Self::get_model_download_url(model))
        .with_quantization(if model.as_str().contains("q5_1") {
            "q5_1".to_string()
        } else if model.as_str().contains("q8_0") {
            "q8_0".to_string()
        } else if model.as_str().contains("q5_0") {
            "q5_0".to_string()
        } else {
            "fp16".to_string()
        })
    }

    /// Helper method to get download URL for a model (since `get_url` is private)
    fn get_model_download_url(model: &WhisperModel) -> String {
        let base_url = if model.as_str().contains("tdrz") {
            "https://huggingface.co/akashmjn/tinydiarize-whisper.cpp/resolve/main"
        } else {
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
        };
        format!("{}/ggml-{}.bin", base_url, model.as_str())
    }
}
