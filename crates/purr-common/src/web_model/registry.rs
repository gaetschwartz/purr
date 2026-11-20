//! Validated, queryable view of the web model manifest.

use crate::language::{support::LanguageSupport, tag::LanguageTag};
use crate::model::{
    descriptor::{ModelDescriptor, Weights},
    engine::EngineId,
    family::ModelFamily,
    format::ModelFormat,
    profile::ModelProfile,
};
use crate::web_model::attributes::OnnxDtype;
use crate::web_model::digest::Sha256Digest;
use crate::web_model::error::{UnknownWebModelError, WebManifestError};
use crate::web_model::file_role::ModelFileRole;
use crate::web_model::manifest::{
    HUGGING_FACE_ORIGIN, SUPPORTED_SCHEMA, WebManifest, WebManifestConversion, WebManifestFile,
    WebManifestModel, WebManifestVariant, file_url,
};
use crate::web_model::variant::{ShaderF16, TableStorage, VariantKind};
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
    /// The ONNX files become a graph and an external weights file while installing.
    pub converted: Option<WebManifestConversion>,
}

/// A file as the browser stores it in its cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedFile {
    /// Repository path the cache entry is keyed by.
    pub path: String,
    pub size_bytes: u64,
    pub sha256: Sha256Digest,
}

impl WebModelFile {
    /// Suffix transformers.js appends to an ONNX path to ask for its external weights.
    pub const DATA_SUFFIX: &'static str = "_data";

    /// The files the cache holds once this one is installed: the downloaded file itself, or for
    /// an ONNX file its converted graph and weights.
    #[must_use]
    pub fn cached_files(&self) -> Vec<CachedFile> {
        match &self.converted {
            None => vec![CachedFile {
                path: self.path.clone(),
                size_bytes: self.size_bytes,
                sha256: self.sha256,
            }],
            Some(converted) => vec![
                CachedFile {
                    path: self.path.clone(),
                    size_bytes: converted.graph.size_bytes,
                    sha256: converted.graph.sha256,
                },
                CachedFile {
                    path: format!("{}{}", self.path, Self::DATA_SUFFIX),
                    size_bytes: converted.data.size_bytes,
                    sha256: converted.data.sha256,
                },
            ],
        }
    }
}

/// One variant of a browser model: everything needed to install and run its weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebModelSpec {
    pub id: String,
    pub repo: String,
    pub revision: String,
    pub family: ModelFamily,
    pub languages: LanguageSupport,
    pub description: String,
    pub variant: VariantKind,
    pub encoder_dtype: OnnxDtype,
    pub decoder_dtype: OnnxDtype,
    pub table: TableStorage,
    /// Every file the pipeline loads, in [`ModelFileRole::ALL`] order.
    pub files: Vec<WebModelFile>,
}

impl WebModelSpec {
    pub const ENGINE: EngineId = EngineId::Browser;
    pub const FORMAT: ModelFormat = ModelFormat::Onnx;

    #[must_use]
    pub fn display_name(&self) -> String {
        let english = if self.english_only() {
            " (English)"
        } else {
            ""
        };
        format!("Whisper {}{english}", self.family.label())
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

    /// Cache key of a stored file.
    #[must_use]
    pub fn cached_url(&self, file: &CachedFile) -> String {
        file_url(&self.repo, &self.revision, &file.path)
    }

    /// Cache keys of everything the cache holds for `file`.
    #[must_use]
    pub fn cached_urls(&self, file: &WebModelFile) -> Vec<String> {
        file.cached_files()
            .iter()
            .map(|cached| self.cached_url(cached))
            .collect()
    }

    #[must_use]
    pub fn repo_url(&self) -> String {
        format!("{HUGGING_FACE_ORIGIN}/{}/tree/{}", self.repo, self.revision)
    }

    #[must_use]
    pub fn english_only(&self) -> bool {
        self.languages
            .single()
            .is_some_and(|tag| *tag == LanguageTag::english())
    }

    /// Whether running this variant on WebGPU needs the `shader-f16` adapter feature.
    #[must_use]
    pub const fn requires_shader_f16(&self) -> bool {
        match self.variant {
            VariantKind::Fp16 => true,
            VariantKind::Fp32 => false,
        }
    }

    /// What the platform knows about the weights of this variant.
    #[must_use]
    pub fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            engine: Self::ENGINE,
            family: self.family,
            format: Self::FORMAT,
            weights: Weights::Onnx {
                encoder: self.encoder_dtype,
                decoder: self.decoder_dtype,
            },
            profile: ModelProfile::browser(self.languages.clone()),
        }
    }

    #[must_use]
    pub const fn runs_on_cpu(&self) -> bool {
        self.encoder_dtype.runs_on_cpu() && self.decoder_dtype.runs_on_cpu()
    }

    /// Files of this variant that none of the `kept` variants stores the same way, so evicting
    /// them leaves every kept variant whole.
    #[must_use]
    pub fn files_not_loaded_by(&self, kept: &[&Self]) -> Vec<&WebModelFile> {
        self.files
            .iter()
            .filter(|file| {
                !kept.iter().any(|other| {
                    other.repo == self.repo
                        && other.revision == self.revision
                        && other.files.iter().any(|shared| {
                            shared.path == file.path
                                && shared.sha256 == file.sha256
                                && shared.converted == file.converted
                        })
                })
            })
            .collect()
    }
}

/// A browser model with every variant it is offered in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebModel {
    pub id: String,
    baseline: WebModelSpec,
    others: Vec<WebModelSpec>,
}

impl WebModel {
    /// Every variant, the baseline first.
    pub fn variants(&self) -> impl Iterator<Item = &WebModelSpec> {
        std::iter::once(&self.baseline).chain(&self.others)
    }

    #[must_use]
    pub fn variant(&self, kind: VariantKind) -> Option<&WebModelSpec> {
        self.variants().find(|spec| spec.variant == kind)
    }

    /// The variant that describes the model to people: the full-precision one, or the only one.
    #[must_use]
    pub const fn baseline(&self) -> &WebModelSpec {
        &self.baseline
    }

    /// The variant to run on WebGPU: half precision where the adapter allows it.
    #[must_use]
    pub fn webgpu_variant(&self, shader_f16: ShaderF16) -> &WebModelSpec {
        let half = match shader_f16 {
            ShaderF16::Supported => self.variant(VariantKind::Fp16),
            ShaderF16::Missing => None,
        };
        half.or_else(|| self.variant(VariantKind::Fp32))
            .unwrap_or_else(|| self.baseline())
    }

    /// The variant to run on the CPU, if any can: full precision first.
    #[must_use]
    pub fn cpu_variant(&self) -> Option<&WebModelSpec> {
        [VariantKind::Fp32, VariantKind::Fp16]
            .into_iter()
            .filter_map(|kind| self.variant(kind))
            .find(|spec| spec.runs_on_cpu())
    }

    /// Whether no variant runs without `shader-f16`.
    #[must_use]
    pub fn requires_shader_f16(&self) -> bool {
        self.variants().all(WebModelSpec::requires_shader_f16)
    }

    #[must_use]
    pub fn runs_on_cpu(&self) -> bool {
        self.cpu_variant().is_some()
    }
}

#[derive(Debug, Clone)]
pub struct WebModelRegistry {
    models: Vec<Arc<WebModel>>,
    default_model: Arc<WebModel>,
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
    pub fn models(&self) -> &[Arc<WebModel>] {
        &self.models
    }

    pub fn get(&self, id: &str) -> Result<&WebModel, UnknownWebModelError> {
        self.models
            .iter()
            .find(|model| model.id == id)
            .map(|model| &**model)
            .ok_or_else(|| UnknownWebModelError {
                model_id: id.to_owned(),
            })
    }

    /// The other models pinned to the same repository revision as `model`; they may load the
    /// same files.
    pub fn siblings<'a>(&'a self, model: &'a WebModel) -> impl Iterator<Item = &'a WebModel> {
        let (repo, revision) = {
            let spec = model.baseline();
            (spec.repo.as_str(), spec.revision.as_str())
        };
        self.models
            .iter()
            .map(|other| &**other)
            .filter(move |other| {
                let spec = other.baseline();
                other.id != model.id && spec.repo == repo && spec.revision == revision
            })
    }

    /// Recommended model for users who have not chosen one.
    #[must_use]
    pub fn default_model(&self) -> &WebModel {
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

/// Checks `files` against the files `expected` and returns them in that order.
fn resolve_files(
    model_id: &str,
    files: &[WebManifestFile],
    expected: &[(ModelFileRole, String)],
) -> Result<Vec<WebModelFile>, WebManifestError> {
    let mut paths = HashSet::new();
    for file in files {
        if !paths.insert(file.path.as_str()) {
            return Err(WebManifestError::DuplicatePath {
                model_id: model_id.to_owned(),
                path: file.path.clone(),
            });
        }
        if file.size_bytes == 0 {
            return Err(WebManifestError::ZeroSize {
                model_id: model_id.to_owned(),
                path: file.path.clone(),
            });
        }
        if file.sha256.is_zero() {
            return Err(WebManifestError::MissingDigest {
                model_id: model_id.to_owned(),
                path: file.path.clone(),
            });
        }
        let Some((role, _)) = expected.iter().find(|(_, path)| *path == file.path) else {
            return Err(WebManifestError::UnexpectedFile {
                model_id: model_id.to_owned(),
                path: file.path.clone(),
            });
        };
        match (&file.converted, role.is_onnx()) {
            (None, true) => {
                return Err(WebManifestError::MissingConversion {
                    model_id: model_id.to_owned(),
                    path: file.path.clone(),
                });
            }
            (Some(_), false) => {
                return Err(WebManifestError::UnexpectedConversion {
                    model_id: model_id.to_owned(),
                    path: file.path.clone(),
                });
            }
            (Some(converted), true) => {
                for stored in [converted.graph, converted.data] {
                    if stored.size_bytes == 0 {
                        return Err(WebManifestError::ZeroSize {
                            model_id: model_id.to_owned(),
                            path: file.path.clone(),
                        });
                    }
                    if stored.sha256.is_zero() {
                        return Err(WebManifestError::MissingDigest {
                            model_id: model_id.to_owned(),
                            path: file.path.clone(),
                        });
                    }
                }
            }
            (None, false) => {}
        }
    }

    expected
        .iter()
        .map(|(role, path)| {
            let file = files
                .iter()
                .find(|file| file.path == *path)
                .ok_or_else(|| WebManifestError::MissingFile {
                    model_id: model_id.to_owned(),
                    path: path.clone(),
                })?;
            Ok(WebModelFile {
                role: *role,
                path: path.clone(),
                size_bytes: file.size_bytes,
                sha256: file.sha256,
                converted: file.converted,
            })
        })
        .collect()
}

fn resolve_variant(
    entry: &WebManifestModel,
    plain: &[WebModelFile],
    variant: &WebManifestVariant,
) -> Result<WebModelSpec, WebManifestError> {
    for dtype in [variant.encoder_dtype, variant.decoder_dtype] {
        if !dtype.runs_on_webgpu() {
            return Err(WebManifestError::UnsupportedDtype {
                model_id: entry.id.clone(),
                dtype,
            });
        }
    }
    let needs_f16 = variant.encoder_dtype.requires_shader_f16()
        || variant.decoder_dtype.requires_shader_f16()
        || !variant.table.is_fp32();
    let fits = match variant.precision {
        VariantKind::Fp16 => true,
        VariantKind::Fp32 => !needs_f16,
    };
    if !fits {
        return Err(WebManifestError::InconsistentVariant {
            model_id: entry.id.clone(),
            kind: variant.precision,
        });
    }

    let expected: Vec<(ModelFileRole, String)> = ModelFileRole::ALL
        .iter()
        .filter(|role| role.is_onnx())
        .map(|&role| {
            (
                role,
                role.path(variant.encoder_dtype, variant.decoder_dtype),
            )
        })
        .collect();
    let onnx = resolve_files(&entry.id, &variant.files, &expected)?;
    let mut files = Vec::with_capacity(ModelFileRole::ALL.len());
    for role in ModelFileRole::ALL {
        let file = plain
            .iter()
            .chain(&onnx)
            .find(|file| file.role == role)
            .ok_or_else(|| WebManifestError::MissingFile {
                model_id: entry.id.clone(),
                path: role.path(variant.encoder_dtype, variant.decoder_dtype),
            })?;
        files.push(file.clone());
    }

    Ok(WebModelSpec {
        id: entry.id.clone(),
        repo: entry.repo.clone(),
        revision: entry.revision.clone(),
        family: entry.family,
        languages: entry.languages.clone(),
        description: entry.description.clone(),
        variant: variant.precision,
        encoder_dtype: variant.encoder_dtype,
        decoder_dtype: variant.decoder_dtype,
        table: variant.table,
        files,
    })
}

fn resolve(entry: WebManifestModel) -> Result<WebModel, WebManifestError> {
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
    if entry.variants.is_empty() {
        return Err(WebManifestError::NoVariants { model_id: entry.id });
    }

    let expected_plain: Vec<(ModelFileRole, String)> = ModelFileRole::ALL
        .iter()
        .filter(|role| !role.is_onnx())
        .map(|&role| (role, role.path(OnnxDtype::Fp32, OnnxDtype::Fp32)))
        .collect();
    let plain = resolve_files(&entry.id, &entry.files, &expected_plain)?;

    let mut kinds = HashSet::new();
    let mut variants = Vec::with_capacity(entry.variants.len());
    for variant in &entry.variants {
        if !kinds.insert(variant.precision) {
            return Err(WebManifestError::DuplicateVariant {
                model_id: entry.id,
                kind: variant.precision,
            });
        }
        variants.push(resolve_variant(&entry, &plain, variant)?);
    }
    variants.sort_by_key(|spec| spec.variant != VariantKind::Fp32);
    let mut ordered = variants.into_iter();
    let baseline = ordered.next().ok_or_else(|| WebManifestError::NoVariants {
        model_id: entry.id.clone(),
    })?;
    Ok(WebModel {
        id: entry.id,
        baseline,
        others: ordered.collect(),
    })
}
