//! Serialized form of the web model manifest.

use crate::language::support::LanguageSupport;
use crate::model::family::ModelFamily;
use crate::web_model::attributes::OnnxDtype;
use crate::web_model::digest::{FileDigest, Sha256Digest};
use crate::web_model::error::WebManifestError;
use crate::web_model::variant::{TableStorage, VariantKind};
use serde::{Deserialize, Serialize};

pub const SUPPORTED_SCHEMA: u32 = 4;
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
    pub family: ModelFamily,
    pub languages: LanguageSupport,
    pub description: String,
    /// The configuration and tokenizer files every variant loads.
    pub files: Vec<WebManifestFile>,
    pub variants: Vec<WebManifestVariant>,
}

/// One set of ONNX weights a model can run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebManifestVariant {
    pub precision: VariantKind,
    pub encoder_dtype: OnnxDtype,
    pub decoder_dtype: OnnxDtype,
    /// How the converter stores the decoder's tied embedding table.
    #[serde(default, skip_serializing_if = "TableStorage::is_fp32")]
    pub table: TableStorage,
    pub files: Vec<WebManifestFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebManifestFile {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: Sha256Digest,
    /// What the browser stores for an ONNX file after the install-time conversion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub converted: Option<WebManifestConversion>,
}

/// The two files a converted ONNX file becomes: the graph and its external weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebManifestConversion {
    pub graph: FileDigest,
    pub data: FileDigest,
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
