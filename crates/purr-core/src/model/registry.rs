//! Validated, queryable view of the model manifest.

use crate::error::{ManifestError, UnknownModelError};
use crate::math::ByteSpeed;
use crate::model::attributes::{ModelFamily, ModelFeature, ModelLanguage, Quantization};
use crate::model::digest::Sha256Digest;
use crate::model::manifest::{Manifest, ManifestModel, SUPPORTED_SCHEMA, Source};
use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

#[cfg(test)]
mod tests;

const EMBEDDED_MANIFEST: &str = include_str!("../../models/manifest.json");
const REVISION_HEX_LEN: usize = 40;

static EMBEDDED: LazyLock<Result<Arc<Registry>, ManifestError>> =
    LazyLock::new(|| Registry::from_json(EMBEDDED_MANIFEST).map(Arc::new));

/// A model that can be downloaded and run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub id: String,
    pub source: Arc<Source>,
    pub filename: String,
    pub family: ModelFamily,
    pub language: ModelLanguage,
    pub quantization: Quantization,
    pub features: Vec<ModelFeature>,
    pub size_bytes: u64,
    pub sha256: Sha256Digest,
    pub description: String,
}

impl ModelSpec {
    #[must_use]
    pub fn url(&self) -> String {
        self.source.file_url(&self.filename)
    }

    #[must_use]
    pub fn has_feature(&self, feature: ModelFeature) -> bool {
        self.features.contains(&feature)
    }

    #[must_use]
    pub fn estimated_download_time(&self, speed: ByteSpeed) -> Duration {
        self.size_bytes as usize / speed
    }
}

/// Models of one family, in manifest order.
#[derive(Debug, Clone)]
pub struct ModelGroup<'a> {
    pub family: ModelFamily,
    pub models: Vec<&'a ModelSpec>,
}

#[derive(Debug, Clone)]
pub struct Registry {
    models: Vec<Arc<ModelSpec>>,
    preferred: Vec<Arc<ModelSpec>>,
    default_model: Arc<ModelSpec>,
}

impl Registry {
    /// The registry built from the manifest embedded in this crate.
    pub fn embedded() -> Result<Arc<Self>, ManifestError> {
        EMBEDDED.clone()
    }

    pub fn from_json(json: &str) -> Result<Self, ManifestError> {
        Self::from_manifest(Manifest::parse(json)?)
    }

    pub fn from_manifest(manifest: Manifest) -> Result<Self, ManifestError> {
        if manifest.schema != SUPPORTED_SCHEMA {
            return Err(ManifestError::UnsupportedSchema {
                found: manifest.schema,
                supported: SUPPORTED_SCHEMA,
            });
        }

        let mut source_ids = HashSet::new();
        let mut sources = Vec::with_capacity(manifest.sources.len());
        for source in manifest.sources {
            if !source_ids.insert(source.id.clone()) {
                return Err(ManifestError::DuplicateSourceId { id: source.id });
            }
            let valid_revision = source.revision.len() == REVISION_HEX_LEN
                && source.revision.bytes().all(|b| b.is_ascii_hexdigit());
            if !valid_revision {
                return Err(ManifestError::InvalidRevision {
                    source_id: source.id,
                    revision: source.revision,
                });
            }
            sources.push(Arc::new(source));
        }

        let mut model_ids = HashSet::new();
        let mut filenames = HashSet::new();
        let mut models = Vec::with_capacity(manifest.models.len());
        for entry in manifest.models {
            if !model_ids.insert(entry.id.clone()) {
                return Err(ManifestError::DuplicateModelId { id: entry.id });
            }
            if !filenames.insert(entry.filename.clone()) {
                return Err(ManifestError::DuplicateFilename {
                    filename: entry.filename,
                });
            }
            models.push(Arc::new(resolve(entry, &sources)?));
        }

        let mut preferred = Vec::with_capacity(manifest.preferred.len());
        for id in manifest.preferred {
            let spec = models
                .iter()
                .find(|model| model.id == id)
                .ok_or(ManifestError::UnknownPreferredModel { id })?;
            preferred.push(Arc::clone(spec));
        }
        let default_model = preferred
            .first()
            .map(Arc::clone)
            .ok_or(ManifestError::NoPreferredModel)?;

        Ok(Self {
            models,
            preferred,
            default_model,
        })
    }

    #[must_use]
    pub fn models(&self) -> &[Arc<ModelSpec>] {
        &self.models
    }

    pub fn get(&self, id: &str) -> Result<&ModelSpec, UnknownModelError> {
        self.models
            .iter()
            .find(|model| model.id == id)
            .map(|model| &**model)
            .ok_or_else(|| UnknownModelError::new(id))
    }

    /// Recommended model for users who have not chosen one.
    #[must_use]
    pub fn default_model(&self) -> &ModelSpec {
        &self.default_model
    }

    /// Models in order of preference when auto-selecting a downloaded one.
    pub fn preferred(&self) -> impl Iterator<Item = &ModelSpec> {
        self.preferred.iter().map(|model| &**model)
    }

    /// Models grouped by family, smallest family first.
    #[must_use]
    pub fn groups(&self) -> Vec<ModelGroup<'_>> {
        let mut by_family: BTreeMap<ModelFamily, Vec<&ModelSpec>> = BTreeMap::new();
        for model in &self.models {
            by_family.entry(model.family).or_default().push(model);
        }
        by_family
            .into_iter()
            .map(|(family, models)| ModelGroup { family, models })
            .collect()
    }
}

fn resolve(entry: ManifestModel, sources: &[Arc<Source>]) -> Result<ModelSpec, ManifestError> {
    let source = sources
        .iter()
        .find(|source| source.id == entry.source)
        .map(Arc::clone)
        .ok_or_else(|| ManifestError::UnknownSource {
            model_id: entry.id.clone(),
            source_id: entry.source.clone(),
        })?;
    if entry.size_bytes == 0 {
        return Err(ManifestError::ZeroSize { model_id: entry.id });
    }
    if entry.sha256.is_zero() {
        return Err(ManifestError::MissingDigest { model_id: entry.id });
    }
    Ok(ModelSpec {
        id: entry.id,
        source,
        filename: entry.filename,
        family: entry.family,
        language: entry.language,
        quantization: entry.quantization,
        features: entry.features,
        size_bytes: entry.size_bytes,
        sha256: entry.sha256,
        description: entry.description,
    })
}
