//! Serialized form of the web model manifest.

use crate::web_model::attributes::{OnnxDtype, WebModelFamily};
use crate::web_model::digest::Sha256Digest;
use crate::web_model::error::WebManifestError;
use serde::{Deserialize, Serialize};

pub const SUPPORTED_SCHEMA: u32 = 1;
pub const HUGGING_FACE_ORIGIN: &str = "https://huggingface.co";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebManifest {
    pub schema: u32,
    /// Id of the recommended model.
    pub default: String,
    pub models: Vec<WebManifestModel>,
}

/// A Hugging Face repository pinned to a commit, loaded by transformers.js.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebManifestModel {
    pub id: String,
    pub repo: String,
    pub revision: String,
    pub family: WebModelFamily,
    pub encoder_dtype: OnnxDtype,
    pub decoder_dtype: OnnxDtype,
    pub description: String,
    pub files: Vec<WebManifestFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebManifestFile {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: Sha256Digest,
}

impl WebManifest {
    pub fn parse(json: &str) -> Result<Self, WebManifestError> {
        serde_json::from_str(json).map_err(|err| WebManifestError::Parse {
            message: err.to_string(),
        })
    }
}

/// Download URL of a file in a repository at a revision.
#[must_use]
pub fn file_url(repo: &str, revision: &str, path: &str) -> String {
    format!("{HUGGING_FACE_ORIGIN}/{repo}/resolve/{revision}/{path}")
}
