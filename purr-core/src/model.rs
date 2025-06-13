//! Model downloading and management functionality

use crate::error::{Result, WhisperError};
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
    /// Tiny quantized Q5_1 (31 MB)
    TinyQ5_1,
    /// Tiny English-only quantized Q5_1 (31 MB)
    TinyEnQ5_1,
    /// Tiny quantized Q8_0 (42 MB)
    TinyQ8_0,
    /// Base model (142 MB, good balance)
    Base,
    /// Base English-only model (142 MB)
    BaseEn,
    /// Base quantized Q5_1 (103 MB)
    BaseQ5_1,
    /// Base English-only quantized Q5_1 (103 MB)
    BaseEnQ5_1,
    /// Base quantized Q8_0 (149 MB)
    BaseQ8_0,
    /// Small model (466 MB)
    Small,
    /// Small English-only model (466 MB)
    SmallEn,
    /// Small English-only TinyDiarize model
    SmallEnTdrz,
    /// Small quantized Q5_1 (340 MB)
    SmallQ5_1,
    /// Small English-only quantized Q5_1 (340 MB)
    SmallEnQ5_1,
    /// Small quantized Q8_0 (488 MB)
    SmallQ8_0,
    /// Medium model (1.5 GB)
    Medium,
    /// Medium English-only model (1.5 GB)
    MediumEn,
    /// Medium quantized Q5_0 (1.1 GB)
    MediumQ5_0,
    /// Medium English-only quantized Q5_0 (1.1 GB)
    MediumEnQ5_0,
    /// Medium quantized Q8_0 (1.6 GB)
    MediumQ8_0,
    /// Large v1 model (3.0 GB)
    LargeV1,
    /// Large v2 model (3.0 GB)
    LargeV2,
    /// Large v2 quantized Q5_0 (2.3 GB)
    LargeV2Q5_0,
    /// Large v2 quantized Q8_0 (3.2 GB)
    LargeV2Q8_0,
    /// Large v3 model (3.0 GB, most accurate)
    LargeV3,
    /// Large v3 quantized Q5_0 (2.3 GB)
    LargeV3Q5_0,
    /// Large v3 Turbo model (1.5 GB, faster)
    LargeV3Turbo,
    /// Large v3 Turbo quantized Q5_0 (1.2 GB)
    LargeV3TurboQ5_0,
    /// Large v3 Turbo quantized Q8_0 (1.6 GB)
    LargeV3TurboQ8_0,
}

impl WhisperModel {
    /// Get the model identifier string used in filenames and URLs
    pub const fn as_str(&self) -> &'static str {
        match self {
            WhisperModel::Tiny => "tiny",
            WhisperModel::TinyEn => "tiny.en",
            WhisperModel::TinyQ5_1 => "tiny-q5_1",
            WhisperModel::TinyEnQ5_1 => "tiny.en-q5_1",
            WhisperModel::TinyQ8_0 => "tiny-q8_0",
            WhisperModel::Base => "base",
            WhisperModel::BaseEn => "base.en",
            WhisperModel::BaseQ5_1 => "base-q5_1",
            WhisperModel::BaseEnQ5_1 => "base.en-q5_1",
            WhisperModel::BaseQ8_0 => "base-q8_0",
            WhisperModel::Small => "small",
            WhisperModel::SmallEn => "small.en",
            WhisperModel::SmallEnTdrz => "small.en-tdrz",
            WhisperModel::SmallQ5_1 => "small-q5_1",
            WhisperModel::SmallEnQ5_1 => "small.en-q5_1",
            WhisperModel::SmallQ8_0 => "small-q8_0",
            WhisperModel::Medium => "medium",
            WhisperModel::MediumEn => "medium.en",
            WhisperModel::MediumQ5_0 => "medium-q5_0",
            WhisperModel::MediumEnQ5_0 => "medium.en-q5_0",
            WhisperModel::MediumQ8_0 => "medium-q8_0",
            WhisperModel::LargeV1 => "large-v1",
            WhisperModel::LargeV2 => "large-v2",
            WhisperModel::LargeV2Q5_0 => "large-v2-q5_0",
            WhisperModel::LargeV2Q8_0 => "large-v2-q8_0",
            WhisperModel::LargeV3 => "large-v3",
            WhisperModel::LargeV3Q5_0 => "large-v3-q5_0",
            WhisperModel::LargeV3Turbo => "large-v3-turbo",
            WhisperModel::LargeV3TurboQ5_0 => "large-v3-turbo-q5_0",
            WhisperModel::LargeV3TurboQ8_0 => "large-v3-turbo-q8_0",
        }
    }

    /// Get the model description
    pub const fn description(&self) -> &'static str {
        match self {
            WhisperModel::Tiny => "Tiny model (39 MB, fastest, lowest accuracy)",
            WhisperModel::TinyEn => "Tiny English-only model (39 MB, fastest for English)",
            WhisperModel::TinyQ5_1 => "Tiny quantized Q5_1 (31 MB, fastest)",
            WhisperModel::TinyEnQ5_1 => "Tiny English-only quantized Q5_1 (31 MB)",
            WhisperModel::TinyQ8_0 => "Tiny quantized Q8_0 (42 MB)",
            WhisperModel::Base => "Base model (142 MB, good balance of speed and accuracy)",
            WhisperModel::BaseEn => "Base English-only model (142 MB, good for English)",
            WhisperModel::BaseQ5_1 => "Base quantized Q5_1 (103 MB)",
            WhisperModel::BaseEnQ5_1 => "Base English-only quantized Q5_1 (103 MB)",
            WhisperModel::BaseQ8_0 => "Base quantized Q8_0 (149 MB)",
            WhisperModel::Small => "Small model (466 MB, good accuracy)",
            WhisperModel::SmallEn => "Small English-only model (466 MB)",
            WhisperModel::SmallEnTdrz => {
                "Small English-only TinyDiarize model (speaker diarization)"
            }
            WhisperModel::SmallQ5_1 => "Small quantized Q5_1 (340 MB)",
            WhisperModel::SmallEnQ5_1 => "Small English-only quantized Q5_1 (340 MB)",
            WhisperModel::SmallQ8_0 => "Small quantized Q8_0 (488 MB)",
            WhisperModel::Medium => "Medium model (1.5 GB, high accuracy)",
            WhisperModel::MediumEn => "Medium English-only model (1.5 GB)",
            WhisperModel::MediumQ5_0 => "Medium quantized Q5_0 (1.1 GB)",
            WhisperModel::MediumEnQ5_0 => "Medium English-only quantized Q5_0 (1.1 GB)",
            WhisperModel::MediumQ8_0 => "Medium quantized Q8_0 (1.6 GB)",
            WhisperModel::LargeV1 => "Large v1 model (3.0 GB, highest accuracy)",
            WhisperModel::LargeV2 => "Large v2 model (3.0 GB, improved accuracy)",
            WhisperModel::LargeV2Q5_0 => "Large v2 quantized Q5_0 (2.3 GB)",
            WhisperModel::LargeV2Q8_0 => "Large v2 quantized Q8_0 (3.2 GB)",
            WhisperModel::LargeV3 => "Large v3 model (3.0 GB, most accurate)",
            WhisperModel::LargeV3Q5_0 => "Large v3 quantized Q5_0 (2.3 GB)",
            WhisperModel::LargeV3Turbo => "Large v3 Turbo model (1.5 GB, faster large model)",
            WhisperModel::LargeV3TurboQ5_0 => "Large v3 Turbo quantized Q5_0 (1.2 GB)",
            WhisperModel::LargeV3TurboQ8_0 => "Large v3 Turbo quantized Q8_0 (1.6 GB)",
        }
    }

    /// Get all available models
    pub const fn all_models() -> &'static [WhisperModel] {
        &[
            WhisperModel::Tiny,
            WhisperModel::TinyEn,
            WhisperModel::TinyQ5_1,
            WhisperModel::TinyEnQ5_1,
            WhisperModel::TinyQ8_0,
            WhisperModel::Base,
            WhisperModel::BaseEn,
            WhisperModel::BaseQ5_1,
            WhisperModel::BaseEnQ5_1,
            WhisperModel::BaseQ8_0,
            WhisperModel::Small,
            WhisperModel::SmallEn,
            WhisperModel::SmallEnTdrz,
            WhisperModel::SmallQ5_1,
            WhisperModel::SmallEnQ5_1,
            WhisperModel::SmallQ8_0,
            WhisperModel::Medium,
            WhisperModel::MediumEn,
            WhisperModel::MediumQ5_0,
            WhisperModel::MediumEnQ5_0,
            WhisperModel::MediumQ8_0,
            WhisperModel::LargeV1,
            WhisperModel::LargeV2,
            WhisperModel::LargeV2Q5_0,
            WhisperModel::LargeV2Q8_0,
            WhisperModel::LargeV3,
            WhisperModel::LargeV3Q5_0,
            WhisperModel::LargeV3Turbo,
            WhisperModel::LargeV3TurboQ5_0,
            WhisperModel::LargeV3TurboQ8_0,
        ]
    }

    pub const fn size(&self) -> u64 {
        match self {
            WhisperModel::Tiny => 39 * 1024 * 1024,        // 39 MB
            WhisperModel::TinyEn => 39 * 1024 * 1024,      // 39 MB
            WhisperModel::TinyQ5_1 => 31 * 1024 * 1024,    // 31 MB
            WhisperModel::TinyEnQ5_1 => 31 * 1024 * 1024,  // 31 MB
            WhisperModel::TinyQ8_0 => 42 * 1024 * 1024,    // 42 MB
            WhisperModel::Base => 142 * 1024 * 1024,       // 142 MB
            WhisperModel::BaseEn => 142 * 1024 * 1024,     // 142 MB
            WhisperModel::BaseQ5_1 => 103 * 1024 * 1024,   // 103 MB
            WhisperModel::BaseEnQ5_1 => 103 * 1024 * 1024, // 103 MB
            WhisperModel::BaseQ8_0 => 149 * 1024 * 1024,   // 149 MB
            WhisperModel::Small => 466 * 1024 * 1024,      // 466 MB
            WhisperModel::SmallEn => 466 * 1024 * 1024,    // 466 MB
            // TinyDiarize model size is not specified, assuming similar to SmallEn
            // Adjust if actual size is known
            WhisperModel::SmallEnTdrz => {
                466 * 1024 * 1024 // Assuming ~50MB extra for diarization features
            }
            WhisperModel::SmallQ5_1 => 340 * 1024 * 1024, // 340 MB
            WhisperModel::SmallEnQ5_1 => 340 * 1024 * 1024, // 340 MB
            WhisperModel::SmallQ8_0 => 488 * 1024 * 1024, // 488 MB
            WhisperModel::Medium => 1_500 * 1024 * 1024,  // 1.5 GB
            WhisperModel::MediumEn => 1_500 * 1024 * 1024, // 1.5 GB
            WhisperModel::MediumQ5_0 => 1_100 * 1024 * 1024, // 1.1 GB
            WhisperModel::MediumEnQ5_0 => 1_100 * 1024 * 1024, // 1.1 GB
            WhisperModel::MediumQ8_0 => 1_600 * 1024 * 1024, // 1.6 GB
            WhisperModel::LargeV1 => 3_000 * 1024 * 1024, // 3.0 GB
            WhisperModel::LargeV2 => 3_000 * 1024 * 1024, // 3.0 GB
            WhisperModel::LargeV2Q5_0 => 2_300 * 1024 * 1024, // 2.3 GB
            WhisperModel::LargeV2Q8_0 => 3_200 * 1024 * 1024, // 3.2 GB
            WhisperModel::LargeV3 => 3_000 * 1024 * 1024, // 3.0 GB
            WhisperModel::LargeV3Q5_0 => 2_300 * 1024 * 1024, // 2.3 GB
            WhisperModel::LargeV3Turbo => 1_500 * 1024 * 1024, // 1.5 GB
            WhisperModel::LargeV3TurboQ5_0 => 1_200 * 1024 * 1024, // 1.2 GB
            WhisperModel::LargeV3TurboQ8_0 => 1_600 * 1024 * 1024, // 1.6 GB
        }
    }

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
    pub fn filename(&self) -> String {
        format!("ggml-{}.bin", self.as_str())
    }
}

impl FromStr for WhisperModel {
    type Err = WhisperError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "tiny" => Ok(WhisperModel::Tiny),
            "tiny.en" => Ok(WhisperModel::TinyEn),
            "tiny-q5_1" => Ok(WhisperModel::TinyQ5_1),
            "tiny.en-q5_1" => Ok(WhisperModel::TinyEnQ5_1),
            "tiny-q8_0" => Ok(WhisperModel::TinyQ8_0),
            "base" => Ok(WhisperModel::Base),
            "base.en" => Ok(WhisperModel::BaseEn),
            "base-q5_1" => Ok(WhisperModel::BaseQ5_1),
            "base.en-q5_1" => Ok(WhisperModel::BaseEnQ5_1),
            "base-q8_0" => Ok(WhisperModel::BaseQ8_0),
            "small" => Ok(WhisperModel::Small),
            "small.en" => Ok(WhisperModel::SmallEn),
            "small.en-tdrz" => Ok(WhisperModel::SmallEnTdrz),
            "small-q5_1" => Ok(WhisperModel::SmallQ5_1),
            "small.en-q5_1" => Ok(WhisperModel::SmallEnQ5_1),
            "small-q8_0" => Ok(WhisperModel::SmallQ8_0),
            "medium" => Ok(WhisperModel::Medium),
            "medium.en" => Ok(WhisperModel::MediumEn),
            "medium-q5_0" => Ok(WhisperModel::MediumQ5_0),
            "medium.en-q5_0" => Ok(WhisperModel::MediumEnQ5_0),
            "medium-q8_0" => Ok(WhisperModel::MediumQ8_0),
            "large-v1" => Ok(WhisperModel::LargeV1),
            "large-v2" => Ok(WhisperModel::LargeV2),
            "large-v2-q5_0" => Ok(WhisperModel::LargeV2Q5_0),
            "large-v2-q8_0" => Ok(WhisperModel::LargeV2Q8_0),
            "large-v3" => Ok(WhisperModel::LargeV3),
            "large-v3-q5_0" => Ok(WhisperModel::LargeV3Q5_0),
            "large-v3-turbo" => Ok(WhisperModel::LargeV3Turbo),
            "large-v3-turbo-q5_0" => Ok(WhisperModel::LargeV3TurboQ5_0),
            "large-v3-turbo-q8_0" => Ok(WhisperModel::LargeV3TurboQ8_0),
            _ => Err(WhisperError::Configuration(format!(
                "Unknown Whisper model: {}",
                s
            ))),
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
        let project_dirs = ProjectDirs::from("dev.gaetans", "", "purr").ok_or_else(|| {
            WhisperError::Configuration("Failed to get XDG directories".to_string())
        })?;

        let models_dir = project_dirs.data_dir().join("models");

        Ok(Self { models_dir })
    }

    /// Get the models directory path
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// Ensure the models directory exists
    pub async fn ensure_models_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.models_dir).await.map_err(|e| {
            WhisperError::AudioProcessing(format!("Failed to create models directory: {}", e))
        })?;
        Ok(())
    }

    /// Check if a model is already downloaded
    pub async fn is_model_downloaded(&self, model: WhisperModel) -> bool {
        let model_path = self.get_model_path(model);
        model_path.exists()
    }

    /// Get the full path to a model file
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
            .map_err(|e| WhisperError::Configuration(format!("Failed to download model: {}", e)))?;

        if !response.status().is_success() {
            return Err(WhisperError::Configuration(format!(
                "Failed to download model {}: HTTP {}",
                model.as_str(),
                response.status()
            )));
        }

        // Get content length for progress reporting
        let total_size = response.content_length();

        // Create temporary file first, then rename
        let temp_path = model_path.with_extension("tmp");
        let mut file = fs::File::create(&temp_path).await.map_err(|e| {
            WhisperError::AudioProcessing(format!("Failed to create temporary file: {}", e))
        })?;

        let mut stream = response.bytes_stream();
        let mut downloaded = 0u64;

        use futures_util::StreamExt;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                WhisperError::Configuration(format!("Failed to read download chunk: {}", e))
            })?;

            file.write_all(&chunk).await.map_err(|e| {
                WhisperError::AudioProcessing(format!("Failed to write to file: {}", e))
            })?;

            downloaded += chunk.len() as u64;
            progress_callback(downloaded, total_size);
        }

        file.flush()
            .await
            .map_err(|e| WhisperError::AudioProcessing(format!("Failed to flush file: {}", e)))?;

        drop(file);

        // Rename temporary file to final name
        fs::rename(&temp_path, &model_path).await.map_err(|e| {
            WhisperError::AudioProcessing(format!("Failed to rename downloaded file: {}", e))
        })?;

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
            fs::remove_file(&model_path).await.map_err(|e| {
                WhisperError::AudioProcessing(format!("Failed to delete model file: {}", e))
            })?;
            info!("Deleted model {} from {:?}", model.as_str(), model_path);
        }

        Ok(())
    }

    /// Find a default model file
    pub async fn find_default_model(&self) -> Result<PathBuf> {
        // First try to find a model using the model manager (XDG compliant)
        if let Ok(model_manager) = crate::model::ModelManager::new() {
            if let Some(model_path) = model_manager.find_first_available_model().await {
                return Ok(model_path);
            }
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

        Err(WhisperError::Configuration(
            "No Whisper model found".to_string(),
        ))
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
        assert_eq!(WhisperModel::from_str("base"), Ok(WhisperModel::Base));
        assert_eq!(WhisperModel::from_str("base.en"), Ok(WhisperModel::BaseEn));
        let invalid = WhisperModel::from_str("invalid_model");
        assert!(invalid.is_err(), "Expected an error but got: {:?}", invalid);
    }

    #[test]
    fn test_model_filename() {
        assert_eq!(WhisperModel::Base.filename(), "ggml-base.bin");
        assert_eq!(
            WhisperModel::LargeV3Turbo.filename(),
            "ggml-large-v3-turbo.bin"
        );
    }

    #[test]
    fn test_model_description() {
        assert!(WhisperModel::Base.description().contains("142 MB"));
        assert!(WhisperModel::Tiny.description().contains("fastest"));
    }
}
