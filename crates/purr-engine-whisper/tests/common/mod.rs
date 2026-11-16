#![allow(dead_code, reason = "every test binary uses a part of this module")]

use purr_common::devices::DeviceOrder;
use purr_core::{ModelManager, config::RunConfig};
use std::{fmt, path::PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperModel {
    Base,
    Small,
}

impl WhisperModel {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Small => "small",
        }
    }
}

pub enum MissingModel {
    Whisper { id: &'static str, dir: PathBuf },
    VoiceActivity { dir: PathBuf },
}

impl fmt::Display for MissingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Whisper { id, dir } => write!(
                f,
                "the `{id}` model is missing or incomplete in {}: run `purr models download {id}` or `just download-test-models`",
                dir.display()
            ),
            Self::VoiceActivity { dir } => write!(
                f,
                "the Silero voice activity model is missing or incomplete in {}: run `purr models download base` or `just download-test-models`",
                dir.display()
            ),
        }
    }
}

impl fmt::Debug for MissingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for MissingModel {}

pub type SetupResult<T> = Result<T, Box<dyn std::error::Error>>;

pub fn run_config() -> RunConfig {
    RunConfig::new()
        .with_devices(DeviceOrder::cpu_only())
        .with_threads(1)
}

pub async fn model_path(model: WhisperModel) -> SetupResult<PathBuf> {
    let models = ModelManager::new()?;
    let spec = models.model(model.id())?;
    if !models.is_model_downloaded(spec).await? {
        return Err(MissingModel::Whisper {
            id: model.id(),
            dir: models.models_dir().to_path_buf(),
        }
        .into());
    }
    Ok(models.get_model_path(spec))
}

pub async fn vad_model() -> SetupResult<PathBuf> {
    let models = ModelManager::new()?;
    let path = models.vad_model_path().await?;
    path.ok_or_else(|| {
        MissingModel::VoiceActivity {
            dir: models.models_dir().to_path_buf(),
        }
        .into()
    })
}
