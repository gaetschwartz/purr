//! Serialized form of the model manifest.

use crate::error::ManifestError;
use crate::model::attributes::{ModelFamily, ModelFeature, ModelLanguage, Quantization};
use crate::model::digest::Sha256Digest;
use serde::{Deserialize, Serialize};

pub const SUPPORTED_SCHEMA: u32 = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: u32,
    /// Model ids in order of preference; the first is the recommended default.
    pub preferred: Vec<String>,
    pub sources: Vec<Source>,
    pub models: Vec<ManifestModel>,
    pub vad: ManifestVad,
}

/// A Hugging Face repository pinned to a commit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub id: String,
    pub repo: String,
    pub revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestModel {
    pub id: String,
    pub source: String,
    pub filename: String,
    pub family: ModelFamily,
    pub language: ModelLanguage,
    pub quantization: Quantization,
    pub features: Vec<ModelFeature>,
    pub size_bytes: u64,
    pub sha256: Sha256Digest,
    pub description: String,
}

/// The voice activity model that rides along with the Whisper models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestVad {
    pub id: String,
    pub source: String,
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: Sha256Digest,
}

impl Source {
    pub const ORIGIN: &'static str = "https://huggingface.co";

    #[must_use]
    pub fn file_url(&self, filename: &str) -> String {
        format!(
            "{}/{}/resolve/{}/{filename}",
            Self::ORIGIN,
            self.repo,
            self.revision
        )
    }
}

impl Manifest {
    pub fn parse(json: &str) -> Result<Self, ManifestError> {
        serde_json::from_str(json).map_err(|err| ManifestError::Parse {
            message: err.to_string(),
        })
    }
}
