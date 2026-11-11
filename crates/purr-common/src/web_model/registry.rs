//! Validated, queryable view of the web model manifest.

use crate::web_model::attributes::{OnnxDtype, WebModelFamily};
use crate::web_model::digest::Sha256Digest;
use crate::web_model::error::{UnknownWebModelError, WebManifestError};
use crate::web_model::file_role::ModelFileRole;
use crate::web_model::manifest::{
    HUGGING_FACE_ORIGIN, SUPPORTED_SCHEMA, WebManifest, WebManifestModel, file_url,
};
use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

#[cfg(test)]
mod tests;

const EMBEDDED_MANIFEST: &str = include_str!("../../manifests/web-models.json");
const REVISION_HEX_LEN: usize = 40;

static EMBEDDED: LazyLock<Result<Arc<WebModelRegistry>, WebManifestError>> =
    LazyLock::new(|| WebModelRegistry::from_json(EMBEDDED_MANIFEST).map(Arc::new));

/// A file transformers.js downloads for a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebModelFile {
    pub role: ModelFileRole,
    pub path: String,
    pub size_bytes: u64,
    pub sha256: Sha256Digest,
}

/// A browser model that transformers.js can download and run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebModelSpec {
    pub id: String,
    pub repo: String,
    pub revision: String,
    pub family: WebModelFamily,
    pub encoder_dtype: OnnxDtype,
    pub decoder_dtype: OnnxDtype,
    pub description: String,
    /// Every file the pipeline loads, in [`ModelFileRole::ALL`] order.
    pub files: Vec<WebModelFile>,
}

impl WebModelSpec {
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        self.family.display_name()
    }

    /// Total download size: the sum of every file.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        self.files
            .iter()
            .fold(0, |total, file| total.saturating_add(file.size_bytes))
    }

    #[must_use]
    pub fn file_url(&self, file: &WebModelFile) -> String {
        file_url(&self.repo, &self.revision, &file.path)
    }

    #[must_use]
    pub fn repo_url(&self) -> String {
        format!("{HUGGING_FACE_ORIGIN}/{}/tree/{}", self.repo, self.revision)
    }

    #[must_use]
    pub const fn requires_shader_f16(&self) -> bool {
        self.encoder_dtype.requires_shader_f16() || self.decoder_dtype.requires_shader_f16()
    }
}

#[derive(Debug, Clone)]
pub struct WebModelRegistry {
    models: Vec<Arc<WebModelSpec>>,
    default_model: Arc<WebModelSpec>,
}

impl WebModelRegistry {
    /// The registry built from the manifest embedded in this crate.
    pub fn embedded() -> Result<Arc<Self>, WebManifestError> {
        EMBEDDED.clone()
    }

    pub fn from_json(json: &str) -> Result<Self, WebManifestError> {
        Self::from_manifest(WebManifest::parse(json)?)
    }

    pub fn from_manifest(manifest: WebManifest) -> Result<Self, WebManifestError> {
        if manifest.schema != SUPPORTED_SCHEMA {
            return Err(WebManifestError::UnsupportedSchema {
                found: manifest.schema,
                supported: SUPPORTED_SCHEMA,
            });
        }

        let mut model_ids = HashSet::new();
        let mut models = Vec::with_capacity(manifest.models.len());
        for entry in manifest.models {
            if !model_ids.insert(entry.id.clone()) {
                return Err(WebManifestError::DuplicateModelId { id: entry.id });
            }
            models.push(Arc::new(resolve(entry)?));
        }

        let default_model = models
            .iter()
            .find(|model| model.id == manifest.default)
            .map(Arc::clone)
            .ok_or(WebManifestError::UnknownDefaultModel {
                id: manifest.default,
            })?;

        Ok(Self {
            models,
            default_model,
        })
    }

    #[must_use]
    pub fn models(&self) -> &[Arc<WebModelSpec>] {
        &self.models
    }

    pub fn get(&self, id: &str) -> Result<&WebModelSpec, UnknownWebModelError> {
        self.models
            .iter()
            .find(|model| model.id == id)
            .map(|model| &**model)
            .ok_or_else(|| UnknownWebModelError {
                model_id: id.to_owned(),
            })
    }

    /// Recommended model for users who have not chosen one.
    #[must_use]
    pub fn default_model(&self) -> &WebModelSpec {
        &self.default_model
    }
}

fn is_valid_repo(repo: &str) -> bool {
    repo.split_once('/')
        .is_some_and(|(owner, name)| !owner.is_empty() && !name.is_empty() && !name.contains('/'))
}

fn is_valid_revision(revision: &str) -> bool {
    revision.len() == REVISION_HEX_LEN && revision.bytes().all(|b| b.is_ascii_hexdigit())
}

fn resolve(entry: WebManifestModel) -> Result<WebModelSpec, WebManifestError> {
    if !is_valid_repo(&entry.repo) {
        return Err(WebManifestError::InvalidRepo {
            model_id: entry.id,
            repo: entry.repo,
        });
    }
    if !is_valid_revision(&entry.revision) {
        return Err(WebManifestError::InvalidRevision {
            model_id: entry.id,
            revision: entry.revision,
        });
    }

    let expected: Vec<(ModelFileRole, String)> = ModelFileRole::ALL
        .iter()
        .map(|&role| (role, role.path(entry.encoder_dtype, entry.decoder_dtype)))
        .collect();

    let mut paths = HashSet::new();
    for file in &entry.files {
        if !paths.insert(file.path.as_str()) {
            return Err(WebManifestError::DuplicatePath {
                model_id: entry.id,
                path: file.path.clone(),
            });
        }
        if file.size_bytes == 0 {
            return Err(WebManifestError::ZeroSize {
                model_id: entry.id,
                path: file.path.clone(),
            });
        }
        if file.sha256.is_zero() {
            return Err(WebManifestError::MissingDigest {
                model_id: entry.id,
                path: file.path.clone(),
            });
        }
        if !expected.iter().any(|(_, path)| *path == file.path) {
            return Err(WebManifestError::UnexpectedFile {
                model_id: entry.id,
                path: file.path.clone(),
            });
        }
    }

    let mut files = Vec::with_capacity(expected.len());
    for (role, path) in expected {
        let file = entry
            .files
            .iter()
            .find(|file| file.path == path)
            .ok_or_else(|| WebManifestError::MissingFile {
                model_id: entry.id.clone(),
                path: path.clone(),
            })?;
        files.push(WebModelFile {
            role,
            path,
            size_bytes: file.size_bytes,
            sha256: file.sha256,
        });
    }

    Ok(WebModelSpec {
        id: entry.id,
        repo: entry.repo,
        revision: entry.revision,
        family: entry.family,
        encoder_dtype: entry.encoder_dtype,
        decoder_dtype: entry.decoder_dtype,
        description: entry.description,
        files,
    })
}
