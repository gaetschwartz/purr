//! Model downloading and management functionality

pub mod attributes;
pub mod digest;
pub mod download;
pub mod manifest;
pub mod registry;
pub mod status;

use crate::error::{ConfigurationError, Result, WhisperError};
use crate::math::ByteSpeed;
use core::str;
use directories::ProjectDirs;
use reqwest;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info};

/// Available Whisper model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WhisperModel {
    /// Tiny model (39 MB, fastest)
    Tiny,
    /// Tiny English-only model (39 MB)
    TinyEn,
    /// Tiny quantized `Q5_1` (31 MB)
    TinyQ5_1,
    /// Tiny English-only quantized `Q5_1` (31 MB)
    TinyEnQ5_1,
    /// Tiny quantized `Q8_0` (42 MB)
    TinyQ8_0,
    /// Base model (142 MB, good balance)
    Base,
    /// Base English-only model (142 MB)
    BaseEn,
    /// Base quantized `Q5_1` (103 MB)
    BaseQ5_1,
    /// Base English-only quantized `Q5_1` (103 MB)
    BaseEnQ5_1,
    /// Base quantized `Q8_0` (149 MB)
    BaseQ8_0,
    /// Small model (466 MB)
    Small,
    /// Small English-only model (466 MB)
    SmallEn,
    /// Small English-only `TinyDiarize` model
    SmallEnTdrz,
    /// Small quantized `Q5_1` (340 MB)
    SmallQ5_1,
    /// Small English-only quantized `Q5_1` (340 MB)
    SmallEnQ5_1,
    /// Small quantized `Q8_0` (488 MB)
    SmallQ8_0,
    /// Medium model (1.5 GB)
    Medium,
    /// Medium English-only model (1.5 GB)
    MediumEn,
    /// Medium quantized `Q5_0` (1.1 GB)
    MediumQ5_0,
    /// Medium English-only quantized `Q5_0` (1.1 GB)
    MediumEnQ5_0,
    /// Medium quantized `Q8_0` (1.6 GB)
    MediumQ8_0,
    /// Large v1 model (3.0 GB)
    LargeV1,
    /// Large v2 model (3.0 GB)
    LargeV2,
    /// Large v2 quantized `Q5_0` (2.3 GB)
    LargeV2Q5_0,
    /// Large v2 quantized `Q8_0` (3.2 GB)
    LargeV2Q8_0,
    /// Large v3 model (3.0 GB, most accurate)
    LargeV3,
    /// Large v3 quantized `Q5_0` (2.3 GB)
    LargeV3Q5_0,
    /// Large v3 Turbo model (1.5 GB, faster)
    LargeV3Turbo,
    /// Large v3 Turbo quantized `Q5_0` (1.2 GB)
    LargeV3TurboQ5_0,
    /// Large v3 Turbo quantized `Q8_0` (1.6 GB)
    LargeV3TurboQ8_0,
}

impl WhisperModel {
    /// Get the model identifier string used in filenames and URLs
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::TinyEn => "tiny.en",
            Self::TinyQ5_1 => "tiny-q5_1",
            Self::TinyEnQ5_1 => "tiny.en-q5_1",
            Self::TinyQ8_0 => "tiny-q8_0",
            Self::Base => "base",
            Self::BaseEn => "base.en",
            Self::BaseQ5_1 => "base-q5_1",
            Self::BaseEnQ5_1 => "base.en-q5_1",
            Self::BaseQ8_0 => "base-q8_0",
            Self::Small => "small",
            Self::SmallEn => "small.en",
            Self::SmallEnTdrz => "small.en-tdrz",
            Self::SmallQ5_1 => "small-q5_1",
            Self::SmallEnQ5_1 => "small.en-q5_1",
            Self::SmallQ8_0 => "small-q8_0",
            Self::Medium => "medium",
            Self::MediumEn => "medium.en",
            Self::MediumQ5_0 => "medium-q5_0",
            Self::MediumEnQ5_0 => "medium.en-q5_0",
            Self::MediumQ8_0 => "medium-q8_0",
            Self::LargeV1 => "large-v1",
            Self::LargeV2 => "large-v2",
            Self::LargeV2Q5_0 => "large-v2-q5_0",
            Self::LargeV2Q8_0 => "large-v2-q8_0",
            Self::LargeV3 => "large-v3",
            Self::LargeV3Q5_0 => "large-v3-q5_0",
            Self::LargeV3Turbo => "large-v3-turbo",
            Self::LargeV3TurboQ5_0 => "large-v3-turbo-q5_0",
            Self::LargeV3TurboQ8_0 => "large-v3-turbo-q8_0",
        }
    }

    /// Get the model description
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Tiny => "Tiny model (39 MB, fastest, lowest accuracy)",
            Self::TinyEn => "Tiny English-only model (39 MB, fastest for English)",
            Self::TinyQ5_1 => "Tiny quantized Q5_1 (31 MB, fastest)",
            Self::TinyEnQ5_1 => "Tiny English-only quantized Q5_1 (31 MB)",
            Self::TinyQ8_0 => "Tiny quantized Q8_0 (42 MB)",
            Self::Base => "Base model (142 MB, good balance of speed and accuracy)",
            Self::BaseEn => "Base English-only model (142 MB, good for English)",
            Self::BaseQ5_1 => "Base quantized Q5_1 (103 MB)",
            Self::BaseEnQ5_1 => "Base English-only quantized Q5_1 (103 MB)",
            Self::BaseQ8_0 => "Base quantized Q8_0 (149 MB)",
            Self::Small => "Small model (466 MB, good accuracy)",
            Self::SmallEn => "Small English-only model (466 MB)",
            Self::SmallEnTdrz => "Small English-only TinyDiarize model (speaker diarization)",
            Self::SmallQ5_1 => "Small quantized Q5_1 (340 MB)",
            Self::SmallEnQ5_1 => "Small English-only quantized Q5_1 (340 MB)",
            Self::SmallQ8_0 => "Small quantized Q8_0 (488 MB)",
            Self::Medium => "Medium model (1.5 GB, high accuracy)",
            Self::MediumEn => "Medium English-only model (1.5 GB)",
            Self::MediumQ5_0 => "Medium quantized Q5_0 (1.1 GB)",
            Self::MediumEnQ5_0 => "Medium English-only quantized Q5_0 (1.1 GB)",
            Self::MediumQ8_0 => "Medium quantized Q8_0 (1.6 GB)",
            Self::LargeV1 => "Large v1 model (3.0 GB, highest accuracy)",
            Self::LargeV2 => "Large v2 model (3.0 GB, improved accuracy)",
            Self::LargeV2Q5_0 => "Large v2 quantized Q5_0 (2.3 GB)",
            Self::LargeV2Q8_0 => "Large v2 quantized Q8_0 (3.2 GB)",
            Self::LargeV3 => "Large v3 model (3.0 GB, most accurate)",
            Self::LargeV3Q5_0 => "Large v3 quantized Q5_0 (2.3 GB)",
            Self::LargeV3Turbo => "Large v3 Turbo model (1.5 GB, faster large model)",
            Self::LargeV3TurboQ5_0 => "Large v3 Turbo quantized Q5_0 (1.2 GB)",
            Self::LargeV3TurboQ8_0 => "Large v3 Turbo quantized Q8_0 (1.6 GB)",
        }
    }

    /// Get all available models
    #[must_use]
    pub const fn all_models() -> &'static [Self] {
        &[
            Self::Tiny,
            Self::TinyEn,
            Self::TinyQ5_1,
            Self::TinyEnQ5_1,
            Self::TinyQ8_0,
            Self::Base,
            Self::BaseEn,
            Self::BaseQ5_1,
            Self::BaseEnQ5_1,
            Self::BaseQ8_0,
            Self::Small,
            Self::SmallEn,
            Self::SmallEnTdrz,
            Self::SmallQ5_1,
            Self::SmallEnQ5_1,
            Self::SmallQ8_0,
            Self::Medium,
            Self::MediumEn,
            Self::MediumQ5_0,
            Self::MediumEnQ5_0,
            Self::MediumQ8_0,
            Self::LargeV1,
            Self::LargeV2,
            Self::LargeV2Q5_0,
            Self::LargeV2Q8_0,
            Self::LargeV3,
            Self::LargeV3Q5_0,
            Self::LargeV3Turbo,
            Self::LargeV3TurboQ5_0,
            Self::LargeV3TurboQ8_0,
        ]
    }

    #[must_use]
    pub const fn size(&self) -> u64 {
        match self {
            Self::Tiny => 39 * 1024 * 1024,        // 39 MB
            Self::TinyEn => 39 * 1024 * 1024,      // 39 MB
            Self::TinyQ5_1 => 31 * 1024 * 1024,    // 31 MB
            Self::TinyEnQ5_1 => 31 * 1024 * 1024,  // 31 MB
            Self::TinyQ8_0 => 42 * 1024 * 1024,    // 42 MB
            Self::Base => 142 * 1024 * 1024,       // 142 MB
            Self::BaseEn => 142 * 1024 * 1024,     // 142 MB
            Self::BaseQ5_1 => 103 * 1024 * 1024,   // 103 MB
            Self::BaseEnQ5_1 => 103 * 1024 * 1024, // 103 MB
            Self::BaseQ8_0 => 149 * 1024 * 1024,   // 149 MB
            Self::Small => 466 * 1024 * 1024,      // 466 MB
            Self::SmallEn => 466 * 1024 * 1024,    // 466 MB
            // TinyDiarize model size is not specified, assuming similar to SmallEn
            // Adjust if actual size is known
            Self::SmallEnTdrz => {
                466 * 1024 * 1024 // Assuming ~50MB extra for diarization features
            }
            Self::SmallQ5_1 => 340 * 1024 * 1024,      // 340 MB
            Self::SmallEnQ5_1 => 340 * 1024 * 1024,    // 340 MB
            Self::SmallQ8_0 => 488 * 1024 * 1024,      // 488 MB
            Self::Medium => 1_500 * 1024 * 1024,       // 1.5 GB
            Self::MediumEn => 1_500 * 1024 * 1024,     // 1.5 GB
            Self::MediumQ5_0 => 1_100 * 1024 * 1024,   // 1.1 GB
            Self::MediumEnQ5_0 => 1_100 * 1024 * 1024, // 1.1 GB
            Self::MediumQ8_0 => 1_600 * 1024 * 1024,   // 1.6 GB
            Self::LargeV1 => 3_000 * 1024 * 1024,      // 3.0 GB
            Self::LargeV2 => 3_000 * 1024 * 1024,      // 3.0 GB
            Self::LargeV2Q5_0 => 2_300 * 1024 * 1024,  // 2.3 GB
            Self::LargeV2Q8_0 => 3_200 * 1024 * 1024,  // 3.2 GB
            Self::LargeV3 => 3_000 * 1024 * 1024,      // 3.0 GB
            Self::LargeV3Q5_0 => 2_300 * 1024 * 1024,  // 2.3 GB
            Self::LargeV3Turbo => 1_500 * 1024 * 1024, // 1.5 GB
            Self::LargeV3TurboQ5_0 => 1_200 * 1024 * 1024, // 1.2 GB
            Self::LargeV3TurboQ8_0 => 1_600 * 1024 * 1024, // 1.6 GB
        }
    }

    #[must_use]
    pub fn estimated_download_time(&self, speed: ByteSpeed) -> Duration {
        // Calculate time in seconds
        self.size() as usize / speed
    }

    /// Get the download URL for this model
    fn get_url(&self) -> String {
        let base_url = if self.as_str().contains("tdrz") {
            "https://huggingface.co/akashmjn/tinydiarize-whisper.cpp/resolve/main"
        } else {
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
        };

        format!("{}/ggml-{}.bin", base_url, self.as_str())
    }

    /// Get the filename for this model
    #[must_use]
    pub fn filename(&self) -> String {
        format!("ggml-{}.bin", self.as_str())
    }
}

impl FromStr for WhisperModel {
    type Err = WhisperError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "tiny" => Ok(Self::Tiny),
            "tiny.en" => Ok(Self::TinyEn),
            "tiny-q5_1" => Ok(Self::TinyQ5_1),
            "tiny.en-q5_1" => Ok(Self::TinyEnQ5_1),
            "tiny-q8_0" => Ok(Self::TinyQ8_0),
            "base" => Ok(Self::Base),
            "base.en" => Ok(Self::BaseEn),
            "base-q5_1" => Ok(Self::BaseQ5_1),
            "base.en-q5_1" => Ok(Self::BaseEnQ5_1),
            "base-q8_0" => Ok(Self::BaseQ8_0),
            "small" => Ok(Self::Small),
            "small.en" => Ok(Self::SmallEn),
            "small.en-tdrz" => Ok(Self::SmallEnTdrz),
            "small-q5_1" => Ok(Self::SmallQ5_1),
            "small.en-q5_1" => Ok(Self::SmallEnQ5_1),
            "small-q8_0" => Ok(Self::SmallQ8_0),
            "medium" => Ok(Self::Medium),
            "medium.en" => Ok(Self::MediumEn),
            "medium-q5_0" => Ok(Self::MediumQ5_0),
            "medium.en-q5_0" => Ok(Self::MediumEnQ5_0),
            "medium-q8_0" => Ok(Self::MediumQ8_0),
            "large-v1" => Ok(Self::LargeV1),
            "large-v2" => Ok(Self::LargeV2),
            "large-v2-q5_0" => Ok(Self::LargeV2Q5_0),
            "large-v2-q8_0" => Ok(Self::LargeV2Q8_0),
            "large-v3" => Ok(Self::LargeV3),
            "large-v3-q5_0" => Ok(Self::LargeV3Q5_0),
            "large-v3-turbo" => Ok(Self::LargeV3Turbo),
            "large-v3-turbo-q5_0" => Ok(Self::LargeV3TurboQ5_0),
            "large-v3-turbo-q8_0" => Ok(Self::LargeV3TurboQ8_0),
            _ => Err(WhisperError::from(ConfigurationError::unknown_model(s))),
        }
    }
}

/// Model manager for downloading and managing Whisper models
pub struct ModelManager {
    models_dir: PathBuf,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new() -> Result<Self> {
        let project_dirs = ProjectDirs::from("dev.gaetans", "", "purr")
            .ok_or_else(|| WhisperError::from(ConfigurationError::XdgDirectories))?;

        let models_dir = project_dirs.data_dir().join("models");

        Ok(Self { models_dir })
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

    /// Check if a model is already downloaded
    pub async fn is_model_downloaded(&self, model: WhisperModel) -> bool {
        let model_path = self.get_model_path(model);
        model_path.exists()
    }

    /// Get the full path to a model file
    #[must_use]
    pub fn get_model_path(&self, model: WhisperModel) -> PathBuf {
        self.models_dir.join(model.filename())
    }

    /// Assign a model path to a transcription configuration
    pub fn assign_model_path(
        &self,
        config: &mut crate::config::TranscriptionConfig,
        model: WhisperModel,
    ) {
        config.model_path = Some(self.get_model_path(model));
    }

    /// Find the first available model (for auto-selection)
    pub async fn find_first_available_model(&self) -> Option<PathBuf> {
        // Check recommended models in order of preference
        let recommended = [
            WhisperModel::Base,
            WhisperModel::BaseEn,
            WhisperModel::Small,
            WhisperModel::SmallEn,
            WhisperModel::Tiny,
            WhisperModel::TinyEn,
        ];

        for model in recommended {
            if self.is_model_downloaded(model).await {
                return Some(self.get_model_path(model));
            }
        }

        // Check any available model
        for &model in WhisperModel::all_models() {
            if self.is_model_downloaded(model).await {
                return Some(self.get_model_path(model));
            }
        }

        None
    }

    /// Download a model
    pub async fn download_model(&self, model: WhisperModel) -> Result<PathBuf> {
        self.download_model_with_progress(model, |_, _| {}).await
    }

    /// Download a model with progress callback
    pub async fn download_model_with_progress<F>(
        &self,
        model: WhisperModel,
        mut progress_callback: F,
    ) -> Result<PathBuf>
    where
        F: FnMut(u64, Option<u64>), // (downloaded_bytes, total_bytes)
    {
        self.ensure_models_dir().await?;

        let model_path = self.get_model_path(model);

        debug!("Downloading model {} to {:?}", model.as_str(), model_path);

        let url = model.get_url();
        let response = reqwest::get(&url)
            .await
            .map_err(|e| WhisperError::from(ConfigurationError::DownloadFailed { source: e }))?;

        if !response.status().is_success() {
            return Err(WhisperError::from(ConfigurationError::DownloadHttpError {
                model_name: model.as_str().to_owned(),
                status_code: response.status().as_u16(),
            }));
        }

        // Get content length for progress reporting
        let total_size = response.content_length();

        // Create temporary file first, then rename
        let temp_path = model_path.with_extension("tmp");
        let mut file = fs::File::create(&temp_path)
            .await
            .map_err(WhisperError::from)?;

        let mut stream = response.bytes_stream();
        let mut downloaded = 0u64;

        use futures_util::StreamExt;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                WhisperError::from(ConfigurationError::DownloadStreamError { source: e })
            })?;

            file.write_all(&chunk).await.map_err(WhisperError::from)?;

            downloaded += chunk.len() as u64;
            progress_callback(downloaded, total_size);
        }

        file.flush().await.map_err(WhisperError::from)?;

        drop(file);

        // Rename temporary file to final name
        fs::rename(&temp_path, &model_path)
            .await
            .map_err(WhisperError::from)?;

        debug!(
            "Successfully downloaded model {} to {:?}",
            model.as_str(),
            model_path
        );
        Ok(model_path)
    }

    /// List all downloaded models
    pub async fn list_downloaded_models(&self) -> Result<Vec<WhisperModel>> {
        if !self.models_dir.exists() {
            return Ok(Vec::new());
        }

        let mut downloaded = Vec::new();

        for &model in WhisperModel::all_models() {
            if self.is_model_downloaded(model).await {
                downloaded.push(model);
            }
        }

        Ok(downloaded)
    }

    /// Delete a downloaded model
    pub async fn delete_model(&self, model: WhisperModel) -> Result<()> {
        let model_path = self.get_model_path(model);

        if model_path.exists() {
            fs::remove_file(&model_path)
                .await
                .map_err(WhisperError::from)?;
            info!("Deleted model {} from {:?}", model.as_str(), model_path);
        }

        Ok(())
    }

    /// Find a default model file
    pub async fn find_default_model(&self) -> Result<PathBuf> {
        // First try to find a model using the model manager (XDG compliant)
        if let Ok(model_manager) = Self::new()
            && let Some(model_path) = model_manager.find_first_available_model().await
        {
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

impl Default for ModelManager {
    fn default() -> Self {
        Self::new().expect("Failed to create ModelManager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_parsing() {
        // Test successful parsing
        match WhisperModel::from_str("base") {
            Ok(WhisperModel::Base) => {}
            other => panic!("Expected Ok(WhisperModel::Base), got: {other:?}"),
        }

        match WhisperModel::from_str("base.en") {
            Ok(WhisperModel::BaseEn) => {}
            other => panic!("Expected Ok(WhisperModel::BaseEn), got: {other:?}"),
        }

        // Test error case
        let invalid = WhisperModel::from_str("invalid_model");
        assert!(invalid.is_err(), "Expected an error but got: {invalid:?}");
    }
}
