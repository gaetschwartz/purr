//! Errors for the browser model manifest.

/// Error for an invalid or inconsistent web model manifest
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, miette::Diagnostic)]
pub enum WebManifestError {
    #[error("Failed to parse the web model manifest: {message}")]
    #[diagnostic(code(purr::web_manifest::parse))]
    Parse { message: String },

    #[error("Unsupported web model manifest schema {found} (supported: {supported})")]
    #[diagnostic(code(purr::web_manifest::unsupported_schema))]
    UnsupportedSchema { found: u32, supported: u32 },

    #[error("Duplicate web model id in manifest: {id}")]
    #[diagnostic(code(purr::web_manifest::duplicate_model_id))]
    DuplicateModelId { id: String },

    #[error("Web model {model_id} has an invalid repository: {repo}")]
    #[diagnostic(code(purr::web_manifest::invalid_repo))]
    InvalidRepo { model_id: String, repo: String },

    #[error("Web model {model_id} has an invalid revision: {revision}")]
    #[diagnostic(code(purr::web_manifest::invalid_revision))]
    InvalidRevision { model_id: String, revision: String },

    #[error("Web model {model_id} lists {path} twice")]
    #[diagnostic(code(purr::web_manifest::duplicate_path))]
    DuplicatePath { model_id: String, path: String },

    #[error("Web model {model_id} does not list the required file {path}")]
    #[diagnostic(code(purr::web_manifest::missing_file))]
    MissingFile { model_id: String, path: String },

    #[error("Web model {model_id} lists {path}, which transformers.js does not load")]
    #[diagnostic(code(purr::web_manifest::unexpected_file))]
    UnexpectedFile { model_id: String, path: String },

    #[error("Web model {model_id} file {path} has a size of zero bytes")]
    #[diagnostic(code(purr::web_manifest::zero_size))]
    ZeroSize { model_id: String, path: String },

    #[error("Web model {model_id} file {path} has no SHA-256 digest")]
    #[diagnostic(code(purr::web_manifest::missing_digest))]
    MissingDigest { model_id: String, path: String },

    #[error("Default web model {id} is not in the manifest")]
    #[diagnostic(code(purr::web_manifest::unknown_default_model))]
    UnknownDefaultModel { id: String },
}

/// Error for an unknown web model id
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, miette::Diagnostic)]
#[error("Unknown web model: {model_id}")]
#[diagnostic(code(purr::web_manifest::unknown_model))]
pub struct UnknownWebModelError {
    pub model_id: String,
}
