//! Errors for the browser model manifest.

use crate::web_model::attributes::OnnxDtype;
use crate::web_model::variant::VariantKind;

/// Error for an invalid or inconsistent web model manifest
#[oopsie::oopsie]
#[derive(Clone, PartialEq, Eq)]
pub enum WebManifestError {
    #[oopsie(
        "Failed to parse the web model manifest: {message}",
        code = "purr::web_manifest::parse"
    )]
    Parse { message: String },

    #[oopsie(
        "Unsupported web model manifest schema {found} (supported: {supported})",
        code = "purr::web_manifest::unsupported_schema"
    )]
    UnsupportedSchema { found: u32, supported: u32 },

    #[oopsie(
        "Duplicate web model id in manifest: {id}",
        code = "purr::web_manifest::duplicate_model_id"
    )]
    DuplicateModelId { id: String },

    #[oopsie(
        "Web model {model_id} has an invalid repository: {repo}",
        code = "purr::web_manifest::invalid_repo"
    )]
    InvalidRepo { model_id: String, repo: String },

    #[oopsie(
        "Web model {model_id} has an invalid revision: {revision}",
        code = "purr::web_manifest::invalid_revision"
    )]
    InvalidRevision { model_id: String, revision: String },

    #[oopsie(
        "Web model {model_id} lists {path} twice",
        code = "purr::web_manifest::duplicate_path"
    )]
    DuplicatePath { model_id: String, path: String },

    #[oopsie(
        "Web model {model_id} does not list the required file {path}",
        code = "purr::web_manifest::missing_file"
    )]
    MissingFile { model_id: String, path: String },

    #[oopsie(
        "Web model {model_id} lists {path}, which transformers.js does not load",
        code = "purr::web_manifest::unexpected_file"
    )]
    UnexpectedFile { model_id: String, path: String },

    #[oopsie(
        "Web model {model_id} file {path} has a size of zero bytes",
        code = "purr::web_manifest::zero_size"
    )]
    ZeroSize { model_id: String, path: String },

    #[oopsie(
        "Web model {model_id} file {path} has no SHA-256 digest",
        code = "purr::web_manifest::missing_digest"
    )]
    MissingDigest { model_id: String, path: String },

    #[oopsie(
        "Web model {model_id} file {path} is an ONNX model but lists no converted files",
        code = "purr::web_manifest::missing_conversion"
    )]
    MissingConversion { model_id: String, path: String },

    #[oopsie(
        "Web model {model_id} file {path} is not an ONNX model but lists converted files",
        code = "purr::web_manifest::unexpected_conversion"
    )]
    UnexpectedConversion { model_id: String, path: String },

    #[oopsie(
        "Web model {model_id} uses {dtype}, which the WebGPU execution provider cannot run",
        code = "purr::web_manifest::unsupported_dtype"
    )]
    UnsupportedDtype { model_id: String, dtype: OnnxDtype },

    #[oopsie(
        "Web model {model_id} lists no variants",
        code = "purr::web_manifest::no_variants"
    )]
    NoVariants { model_id: String },

    #[oopsie(
        "Web model {model_id} lists the {kind} variant twice",
        code = "purr::web_manifest::duplicate_variant"
    )]
    DuplicateVariant { model_id: String, kind: VariantKind },

    #[oopsie(
        "Web model {model_id} has a {kind} variant whose weights do not fit it",
        code = "purr::web_manifest::inconsistent_variant"
    )]
    InconsistentVariant { model_id: String, kind: VariantKind },

    #[oopsie(
        "Default web model {id} is not in the manifest",
        code = "purr::web_manifest::unknown_default_model"
    )]
    UnknownDefaultModel { id: String },
}

/// Error for an unknown web model id
#[oopsie::oopsie]
#[derive(Clone, PartialEq, Eq)]
#[oopsie(
    "Unknown web model: {model_id}",
    code = "purr::web_manifest::unknown_model"
)]
pub struct UnknownWebModelError {
    pub model_id: String,
}
