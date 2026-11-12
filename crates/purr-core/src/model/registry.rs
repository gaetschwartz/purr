//! Validated, queryable view of the model manifest.

use crate::error::{ManifestError, UnknownModelError};
use crate::math::ByteSpeed;
use crate::model::attributes::{ModelFamily, ModelFeature, ModelLanguage, Quantization};
use crate::model::digest::Sha256Digest;
use crate::model::manifest::{Manifest, ManifestModel, ManifestVad, SUPPORTED_SCHEMA, Source};
use crate::model::pinned::PinnedFile;
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
    pub file: PinnedFile,
    pub description: String,
}

impl ModelSpec {
    #[must_use]
    pub fn url(&self) -> &str {
        &self.file.url
    }

    #[must_use]
    pub fn has_feature(&self, feature: ModelFeature) -> bool {
        self.features.contains(&feature)
    }

    #[must_use]
    pub fn estimated_download_time(&self, speed: ByteSpeed) -> Duration {
        self.file.size_bytes as usize / speed
    }
}

/// The voice activity model that rides along with the Whisper models; it is not one of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VadSpec {
    pub id: String,
    pub source: Arc<Source>,
    pub filename: String,
    pub file: PinnedFile,
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
    vad: VadSpec,
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

        let vad = resolve_vad(manifest.vad, &sources)?;
        if !model_ids.insert(vad.id.clone()) {
            return Err(ManifestError::DuplicateModelId { id: vad.id });
        }
        if !filenames.insert(vad.filename.clone()) {
            return Err(ManifestError::DuplicateFilename {
                filename: vad.filename,
            });
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
            vad,
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

    pub fn get_shared(&self, id: &str) -> Result<Arc<ModelSpec>, UnknownModelError> {
        self.models
            .iter()
            .find(|model| model.id == id)
            .map(Arc::clone)
            .ok_or_else(|| UnknownModelError::new(id))
    }

    /// The voice activity model that goes with the Whisper models.
    #[must_use]
    pub fn vad(&self) -> &VadSpec {
        &self.vad
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

fn pin(
    id: &str,
    source_id: &str,
    filename: &str,
    size_bytes: u64,
    sha256: Sha256Digest,
    sources: &[Arc<Source>],
) -> Result<(Arc<Source>, PinnedFile), ManifestError> {
    let source = sources
        .iter()
        .find(|source| source.id == source_id)
        .map(Arc::clone)
        .ok_or_else(|| ManifestError::UnknownSource {
            model_id: id.to_owned(),
            source_id: source_id.to_owned(),
        })?;
    if size_bytes == 0 {
        return Err(ManifestError::ZeroSize {
            model_id: id.to_owned(),
        });
    }
    if sha256.is_zero() {
        return Err(ManifestError::MissingDigest {
            model_id: id.to_owned(),
        });
    }
    let file = PinnedFile {
        name: id.to_owned(),
        url: source.file_url(filename),
        size_bytes,
        sha256,
    };
    Ok((source, file))
}

fn resolve(entry: ManifestModel, sources: &[Arc<Source>]) -> Result<ModelSpec, ManifestError> {
    let (source, file) = pin(
        &entry.id,
        &entry.source,
        &entry.filename,
        entry.size_bytes,
        entry.sha256,
        sources,
    )?;
    Ok(ModelSpec {
        id: entry.id,
        source,
        filename: entry.filename,
        family: entry.family,
        language: entry.language,
        quantization: entry.quantization,
        features: entry.features,
        file,
        description: entry.description,
    })
}

fn resolve_vad(entry: ManifestVad, sources: &[Arc<Source>]) -> Result<VadSpec, ManifestError> {
    let (source, file) = pin(
        &entry.id,
        &entry.source,
        &entry.filename,
        entry.size_bytes,
        entry.sha256,
        sources,
    )?;
    Ok(VadSpec {
        id: entry.id,
        source,
        filename: entry.filename,
        file,
    })
}
