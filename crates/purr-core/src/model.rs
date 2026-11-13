//! Model downloading and management functionality

pub mod attributes;
pub mod digest;
pub mod download;
pub mod manifest;
pub mod pinned;
pub mod registry;
pub mod status;

use crate::error::{ConfigurationError, Result, WhisperError};
use crate::model::digest::Sha256Hasher;
use crate::model::pinned::PinnedFile;
use crate::model::registry::{ModelSpec, Registry};
use crate::model::status::{ModelFileStatus, ModelIntegrity};
use directories::ProjectDirs;
use purr_common::hugging_face_token::HuggingFaceToken;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs;
use tracing::{debug, info, warn};

const HASH_BUFFER_BYTES: usize = 1 << 20;
const HASH_REPORT_INTERVAL: Duration = Duration::from_millis(100);

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
        status_of(&self.get_model_path(model), &model.file).await
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
        self.verify_model_with_progress(model, |_, _| {}).await
    }

    /// Like [`Self::verify_model`], reporting `(bytes_hashed, total_bytes)` while the file is read
    pub async fn verify_model_with_progress<F>(
        &self,
        model: &ModelSpec,
        progress: F,
    ) -> Result<ModelIntegrity>
    where
        F: FnMut(u64, u64) + Send + 'static,
    {
        integrity_of(&self.get_model_path(model), &model.file, progress).await
    }

    /// Where the voice activity model lives, whether or not it is there yet
    #[must_use]
    pub fn get_vad_path(&self) -> PathBuf {
        self.models_dir.join(&self.registry.vad().filename)
    }

    /// The voice activity model file, only when it is on disk and matches the manifest
    pub async fn vad_model_path(&self) -> Result<Option<PathBuf>> {
        let path = self.get_vad_path();
        Ok(
            match integrity_of(&path, &self.registry.vad().file, |_, _| {}).await? {
                ModelIntegrity::Verified => Some(path),
                ModelIntegrity::Missing
                | ModelIntegrity::SizeMismatch { .. }
                | ModelIntegrity::ChecksumMismatch { .. } => None,
            },
        )
    }

    /// Make sure the voice activity model is on disk and verified, downloading it if it is not
    pub async fn ensure_vad_model(&self, token: Option<&HuggingFaceToken>) -> Result<PathBuf> {
        if let Some(path) = self.vad_model_path().await? {
            return Ok(path);
        }
        self.ensure_models_dir().await?;
        let path = self.get_vad_path();
        let vad = self.registry.vad();
        debug!("Downloading voice activity model {} to {:?}", vad.id, path);
        download::download_verified(&vad.file, token, &path, |_, _| {}).await?;
        Ok(path)
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

        download::download_verified(&model.file, token, &model_path, progress_callback).await?;

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

/// The voice activity model path, or `None` after logging why there is none.
///
/// The model only sharpens a transcription, so a failure to find or fetch it never fails one.
#[must_use]
pub fn vad_or_none(result: Result<Option<PathBuf>>) -> Option<PathBuf> {
    match result {
        Ok(path) => path,
        Err(error) => {
            warn!("Continuing without the voice activity model: {error}");
            None
        }
    }
}

async fn status_of(path: &Path, file: &PinnedFile) -> Result<ModelFileStatus> {
    match fs::metadata(path).await {
        Ok(metadata) if metadata.len() == file.size_bytes => Ok(ModelFileStatus::Present),
        Ok(metadata) => Ok(ModelFileStatus::SizeMismatch {
            expected: file.size_bytes,
            actual: metadata.len(),
        }),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(ModelFileStatus::Missing),
        Err(err) => Err(err.into()),
    }
}

async fn integrity_of<F>(path: &Path, file: &PinnedFile, progress: F) -> Result<ModelIntegrity>
where
    F: FnMut(u64, u64) + Send + 'static,
{
    match status_of(path, file).await? {
        ModelFileStatus::Missing => Ok(ModelIntegrity::Missing),
        ModelFileStatus::SizeMismatch { expected, actual } => {
            Ok(ModelIntegrity::SizeMismatch { expected, actual })
        }
        ModelFileStatus::Present => {
            let actual = hash_file(path.to_path_buf(), file.size_bytes, progress).await?;
            Ok(if actual == file.sha256 {
                ModelIntegrity::Verified
            } else {
                ModelIntegrity::ChecksumMismatch {
                    expected: file.sha256,
                    actual,
                }
            })
        }
    }
}

async fn hash_file<F>(path: PathBuf, total: u64, progress: F) -> Result<digest::Sha256Digest>
where
    F: FnMut(u64, u64) + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let file = std::fs::File::open(path)?;
        hash_reader(file, total, HASH_REPORT_INTERVAL, progress)
    })
    .await?
    .map_err(WhisperError::from)
}

fn hash_reader<R: Read>(
    mut reader: R,
    total: u64,
    interval: Duration,
    mut progress: impl FnMut(u64, u64),
) -> io::Result<digest::Sha256Digest> {
    let mut hasher = Sha256Hasher::new();
    let mut buffer = vec![0u8; HASH_BUFFER_BYTES];
    let mut hashed = 0u64;
    let mut reported = 0u64;
    let mut last_report = Instant::now();
    progress(0, total);
    loop {
        let read = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        };
        hasher.write_all(&buffer[..read])?;
        hashed += read as u64;
        if last_report.elapsed() >= interval {
            progress(hashed, total);
            reported = hashed;
            last_report = Instant::now();
        }
    }
    if reported != hashed {
        progress(hashed, total);
    }
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn hashing_reports_every_chunk_ending_at_the_total() -> io::Result<()> {
        let data = vec![7u8; HASH_BUFFER_BYTES * 3 + 5];
        let total = data.len() as u64;
        let mut seen = Vec::new();
        let digest = hash_reader(Cursor::new(&data), total, Duration::ZERO, |hashed, of| {
            seen.push((hashed, of));
        })?;
        let chunk = HASH_BUFFER_BYTES as u64;
        assert_eq!(
            seen,
            [
                (0, total),
                (chunk, total),
                (chunk * 2, total),
                (chunk * 3, total),
                (total, total)
            ]
        );
        let mut plain = Sha256Hasher::new();
        plain.write_all(&data)?;
        assert_eq!(digest, plain.finalize());
        Ok(())
    }

    #[test]
    fn throttled_hashing_still_reports_the_start_and_the_end() -> io::Result<()> {
        let data = vec![1u8; HASH_BUFFER_BYTES + 1];
        let total = data.len() as u64;
        let mut seen = Vec::new();
        hash_reader(
            Cursor::new(&data),
            total,
            Duration::from_secs(3600),
            |hashed, of| seen.push((hashed, of)),
        )?;
        assert_eq!(seen, [(0, total), (total, total)]);
        Ok(())
    }
}
