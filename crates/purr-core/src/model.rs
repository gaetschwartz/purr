//! Model downloading and management functionality

pub mod attributes;
pub mod digest;
pub mod download;
pub mod manifest;
pub mod registry;
pub mod status;

use crate::error::{ConfigurationError, Result, WhisperError};
use crate::model::digest::Sha256Hasher;
use crate::model::registry::{ModelSpec, Registry};
use crate::model::status::{ModelFileStatus, ModelIntegrity};
use directories::ProjectDirs;
use purr_common::hugging_face_token::HuggingFaceToken;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tracing::{debug, info};

const HASH_BUFFER_BYTES: usize = 1 << 20;

/// Model manager for downloading and managing Whisper models
pub struct ModelManager {
    models_dir: PathBuf,
    registry: Arc<Registry>,
}

impl ModelManager {
    /// Create a new model manager over the embedded manifest
    pub fn new() -> Result<Self> {
        let project_dirs = ProjectDirs::from("dev.gaetans", "", "purr")
            .ok_or_else(|| WhisperError::from(ConfigurationError::XdgDirectories))?;

        Ok(Self::with_registry(
            project_dirs.data_dir().join("models"),
            Registry::embedded()?,
        ))
    }

    #[must_use]
    pub const fn with_registry(models_dir: PathBuf, registry: Arc<Registry>) -> Self {
        Self {
            models_dir,
            registry,
        }
    }

    #[must_use]
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Look a model up by its manifest id
    pub fn model(&self, id: &str) -> Result<&ModelSpec> {
        Ok(self.registry.get(id)?)
    }

    pub fn shared_model(&self, id: &str) -> Result<Arc<ModelSpec>> {
        Ok(self.registry.get_shared(id)?)
    }

    /// Get the models directory path
    #[must_use]
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// Ensure the models directory exists
    pub async fn ensure_models_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.models_dir)
            .await
            .map_err(WhisperError::from)?;
        Ok(())
    }

    /// Get the full path to a model file
    #[must_use]
    pub fn get_model_path(&self, model: &ModelSpec) -> PathBuf {
        self.models_dir.join(&model.filename)
    }

    /// Compare the model file on disk against the manifest size
    pub async fn file_status(&self, model: &ModelSpec) -> Result<ModelFileStatus> {
        match fs::metadata(self.get_model_path(model)).await {
            Ok(metadata) if metadata.len() == model.size_bytes => Ok(ModelFileStatus::Present),
            Ok(metadata) => Ok(ModelFileStatus::SizeMismatch {
                expected: model.size_bytes,
                actual: metadata.len(),
            }),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(ModelFileStatus::Missing),
            Err(err) => Err(err.into()),
        }
    }

    /// Check if a model is downloaded and has the size recorded in the manifest
    pub async fn is_model_downloaded(&self, model: &ModelSpec) -> Result<bool> {
        Ok(match self.file_status(model).await? {
            ModelFileStatus::Present => true,
            ModelFileStatus::Missing | ModelFileStatus::SizeMismatch { .. } => false,
        })
    }

    /// Hash the model file on disk and compare it against the manifest
    pub async fn verify_model(&self, model: &ModelSpec) -> Result<ModelIntegrity> {
        match self.file_status(model).await? {
            ModelFileStatus::Missing => Ok(ModelIntegrity::Missing),
            ModelFileStatus::SizeMismatch { expected, actual } => {
                Ok(ModelIntegrity::SizeMismatch { expected, actual })
            }
            ModelFileStatus::Present => {
                let actual = hash_file(self.get_model_path(model)).await?;
                Ok(if actual == model.sha256 {
                    ModelIntegrity::Verified
                } else {
                    ModelIntegrity::ChecksumMismatch {
                        expected: model.sha256,
                        actual,
                    }
                })
            }
        }
    }

    /// Find the first available model (for auto-selection)
    pub async fn find_first_available_model(&self) -> Result<Option<PathBuf>> {
        let preferred = self.registry.preferred();
        let all = self.registry.models().iter().map(|model| &**model);

        for model in preferred.chain(all) {
            if self.is_model_downloaded(model).await? {
                return Ok(Some(self.get_model_path(model)));
            }
        }

        Ok(None)
    }

    /// Download a model
    pub async fn download_model(
        &self,
        model: &ModelSpec,
        token: Option<&HuggingFaceToken>,
    ) -> Result<PathBuf> {
        self.download_model_with_progress(model, token, |_, _| {})
            .await
    }

    /// Download a model with progress callback
    pub async fn download_model_with_progress<F>(
        &self,
        model: &ModelSpec,
        token: Option<&HuggingFaceToken>,
        progress_callback: F,
    ) -> Result<PathBuf>
    where
        F: FnMut(u64, Option<u64>), // (downloaded_bytes, total_bytes)
    {
        self.ensure_models_dir().await?;

        let model_path = self.get_model_path(model);
        debug!("Downloading model {} to {:?}", model.id, model_path);

        download::download_verified(model, &model.url(), token, &model_path, progress_callback)
            .await?;

        debug!(
            "Successfully downloaded model {} to {:?}",
            model.id, model_path
        );
        Ok(model_path)
    }

    /// Models that have a file on disk, with the state of that file
    pub async fn local_models(&self) -> Result<Vec<(&ModelSpec, ModelFileStatus)>> {
        let mut local = Vec::new();

        for model in self.registry.models() {
            match self.file_status(model).await? {
                ModelFileStatus::Missing => {}
                status @ (ModelFileStatus::Present | ModelFileStatus::SizeMismatch { .. }) => {
                    local.push((&**model, status));
                }
            }
        }

        Ok(local)
    }

    /// List all downloaded models
    pub async fn list_downloaded_models(&self) -> Result<Vec<&ModelSpec>> {
        Ok(self
            .local_models()
            .await?
            .into_iter()
            .filter_map(|(model, status)| match status {
                ModelFileStatus::Present => Some(model),
                ModelFileStatus::Missing | ModelFileStatus::SizeMismatch { .. } => None,
            })
            .collect())
    }

    /// Delete a downloaded model
    pub async fn delete_model(&self, model: &ModelSpec) -> Result<()> {
        let model_path = self.get_model_path(model);

        match fs::remove_file(&model_path).await {
            Ok(()) => info!("Deleted model {} from {:?}", model.id, model_path),
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }

        Ok(())
    }

    /// Find a default model file
    pub async fn find_default_model(&self) -> Result<PathBuf> {
        if let Some(model_path) = self.find_first_available_model().await? {
            return Ok(model_path);
        }

        // Fallback to legacy locations for backward compatibility
        let possible_paths = [
            "models/ggml-base.en.bin",
            "models/ggml-base.bin",
            "ggml-base.en.bin",
            "ggml-base.bin",
            "whisper-base.en.bin",
            "whisper-base.bin",
        ];

        for path in &possible_paths {
            let path_buf = PathBuf::from(path);
            if path_buf.exists() {
                return Ok(path_buf);
            }
        }

        Err(WhisperError::from(ConfigurationError::NoModelFound))
    }
}

async fn hash_file(path: PathBuf) -> Result<digest::Sha256Digest> {
    tokio::task::spawn_blocking(move || {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::with_capacity(HASH_BUFFER_BYTES, file);
        let mut hasher = Sha256Hasher::new();
        io::copy(&mut reader, &mut hasher)?;
        Ok::<_, io::Error>(hasher.finalize())
    })
    .await?
    .map_err(WhisperError::from)
}
