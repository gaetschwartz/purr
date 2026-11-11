use super::*;
use serde_json::{Value, json};

const DIGEST: &str = "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21";
const ZERO_DIGEST: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";

fn files_json(encoder: OnnxDtype, decoder: OnnxDtype) -> Value {
    ModelFileRole::ALL
        .iter()
        .zip(1u64..)
        .map(|(role, size)| {
            json!({
                "path": role.path(encoder, decoder),
                "size_bytes": size * 100,
                "sha256": DIGEST,
            })
        })
        .collect()
}

fn manifest_json(edit: impl FnOnce(&mut Value)) -> String {
    let mut value = json!({
        "schema": 1,
        "default": "alpha",
        "models": [
            {
                "id": "alpha", "repo": "owner/alpha", "revision": REVISION,
                "family": "base", "encoder_dtype": "fp32", "decoder_dtype": "q4",
                "description": "Alpha",
                "files": files_json(OnnxDtype::Fp32, OnnxDtype::Q4),
            },
            {
                "id": "beta", "repo": "owner/beta", "revision": REVISION,
                "family": "large-v3-turbo", "encoder_dtype": "fp16", "decoder_dtype": "q4",
                "description": "Beta",
                "files": files_json(OnnxDtype::Fp16, OnnxDtype::Q4),
            },
        ],
    });
    edit(&mut value);
    value.to_string()
}

fn registry_with(edit: impl FnOnce(&mut Value)) -> Result<WebModelRegistry, WebManifestError> {
    WebModelRegistry::from_json(&manifest_json(edit))
}

#[test]
fn valid_manifest_resolves_every_model() -> Result<(), WebManifestError> {
    let registry = registry_with(|_| {})?;
    let ids: Vec<&str> = registry.models().iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, ["alpha", "beta"]);
    assert_eq!(registry.default_model().id, "alpha");
    Ok(())
}

#[test]
fn attributes_are_typed() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|_| {})?;
    let beta = registry.get("beta")?;
    assert_eq!(beta.family, WebModelFamily::LargeV3Turbo);
    assert_eq!(beta.encoder_dtype, OnnxDtype::Fp16);
    assert_eq!(beta.decoder_dtype, OnnxDtype::Q4);
    Ok(())
}

#[test]
fn files_follow_role_order_and_dtype_paths() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|v| {
        if let Some(files) = v["models"][1]["files"].as_array_mut() {
            files.reverse();
        }
    })?;
    let beta = registry.get("beta")?;
    let layout: Vec<(ModelFileRole, &str)> = beta
        .files
        .iter()
        .map(|file| (file.role, file.path.as_str()))
        .collect();
    assert_eq!(
        layout,
        [
            (ModelFileRole::Config, "config.json"),
            (ModelFileRole::GenerationConfig, "generation_config.json"),
            (
                ModelFileRole::PreprocessorConfig,
                "preprocessor_config.json"
            ),
            (ModelFileRole::Tokenizer, "tokenizer.json"),
            (ModelFileRole::TokenizerConfig, "tokenizer_config.json"),
            (ModelFileRole::Encoder, "onnx/encoder_model_fp16.onnx"),
            (
                ModelFileRole::DecoderMerged,
                "onnx/decoder_model_merged_q4.onnx"
            ),
        ]
    );
    Ok(())
}

#[test]
fn total_size_is_the_sum_of_every_file() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|_| {})?;
    assert_eq!(registry.get("alpha")?.total_size_bytes(), 2800);
    Ok(())
}

#[test]
fn urls_are_built_from_repo_revision_and_path() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|_| {})?;
    let alpha = registry.get("alpha")?;
    let encoder = alpha
        .files
        .iter()
        .find(|file| file.role == ModelFileRole::Encoder)
        .ok_or("no encoder file")?;
    assert_eq!(
        alpha.file_url(encoder),
        format!("https://huggingface.co/owner/alpha/resolve/{REVISION}/onnx/encoder_model.onnx")
    );
    assert_eq!(
        alpha.repo_url(),
        format!("https://huggingface.co/owner/alpha/tree/{REVISION}")
    );
    Ok(())
}

#[test]
fn shader_f16_is_required_by_half_precision_dtypes() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|_| {})?;
    assert!(!registry.get("alpha")?.requires_shader_f16());
    assert!(registry.get("beta")?.requires_shader_f16());
    Ok(())
}

#[test]
fn dtype_file_suffixes_match_transformers_js() {
    let suffixes: Vec<&str> = [
        OnnxDtype::Fp32,
        OnnxDtype::Fp16,
        OnnxDtype::Q8,
        OnnxDtype::Int8,
        OnnxDtype::Uint8,
        OnnxDtype::Q4,
        OnnxDtype::Bnb4,
        OnnxDtype::Q4F16,
    ]
    .iter()
    .map(|dtype| dtype.file_suffix())
    .collect();
    assert_eq!(
        suffixes,
        [
            "",
            "_fp16",
            "_quantized",
            "_int8",
            "_uint8",
            "_q4",
            "_bnb4",
            "_q4f16"
        ]
    );
}

#[test]
fn lookup_of_unknown_id_is_a_typed_error() -> Result<(), WebManifestError> {
    let registry = registry_with(|_| {})?;
    assert_eq!(
        registry.get("gamma").err(),
        Some(UnknownWebModelError {
            model_id: "gamma".to_owned()
        })
    );
    Ok(())
}

#[test]
fn rejects_unsupported_schema() {
    let result = registry_with(|v| v["schema"] = 2.into());
    assert_eq!(
        result.err(),
        Some(WebManifestError::UnsupportedSchema {
            found: 2,
            supported: SUPPORTED_SCHEMA
        })
    );
}

#[test]
fn rejects_unparseable_manifest() {
    let result = WebModelRegistry::from_json("{");
    assert!(matches!(result, Err(WebManifestError::Parse { .. })));
}

#[test]
fn rejects_duplicate_model_id() {
    let result = registry_with(|v| v["models"][1]["id"] = "alpha".into());
    assert_eq!(
        result.err(),
        Some(WebManifestError::DuplicateModelId {
            id: "alpha".to_owned()
        })
    );
}

#[test]
fn rejects_unknown_default_model() {
    let result = registry_with(|v| v["default"] = "gamma".into());
    assert_eq!(
        result.err(),
        Some(WebManifestError::UnknownDefaultModel {
            id: "gamma".to_owned()
        })
    );
}

#[test]
fn rejects_invalid_repo() {
    let result = registry_with(|v| v["models"][0]["repo"] = "no-owner".into());
    assert_eq!(
        result.err(),
        Some(WebManifestError::InvalidRepo {
            model_id: "alpha".to_owned(),
            repo: "no-owner".to_owned()
        })
    );
}

#[test]
fn rejects_branch_name_as_revision() {
    let result = registry_with(|v| v["models"][0]["revision"] = "main".into());
    assert_eq!(
        result.err(),
        Some(WebManifestError::InvalidRevision {
            model_id: "alpha".to_owned(),
            revision: "main".to_owned()
        })
    );
}

#[test]
fn rejects_duplicate_path() {
    let result = registry_with(|v| v["models"][0]["files"][1]["path"] = "config.json".into());
    assert_eq!(
        result.err(),
        Some(WebManifestError::DuplicatePath {
            model_id: "alpha".to_owned(),
            path: "config.json".to_owned()
        })
    );
}

#[test]
fn rejects_zero_size() {
    let result = registry_with(|v| v["models"][0]["files"][0]["size_bytes"] = 0.into());
    assert_eq!(
        result.err(),
        Some(WebManifestError::ZeroSize {
            model_id: "alpha".to_owned(),
            path: "config.json".to_owned()
        })
    );
}

#[test]
fn rejects_zero_digest() {
    let result = registry_with(|v| v["models"][0]["files"][0]["sha256"] = ZERO_DIGEST.into());
    assert_eq!(
        result.err(),
        Some(WebManifestError::MissingDigest {
            model_id: "alpha".to_owned(),
            path: "config.json".to_owned()
        })
    );
}

#[test]
fn rejects_missing_required_file() {
    let result = registry_with(|v| {
        if let Some(files) = v["models"][0]["files"].as_array_mut() {
            files.pop();
        }
    });
    assert_eq!(
        result.err(),
        Some(WebManifestError::MissingFile {
            model_id: "alpha".to_owned(),
            path: "onnx/decoder_model_merged_q4.onnx".to_owned()
        })
    );
}

#[test]
fn rejects_file_that_does_not_match_the_declared_dtype() {
    let result = registry_with(|v| v["models"][0]["encoder_dtype"] = "fp16".into());
    assert_eq!(
        result.err(),
        Some(WebManifestError::UnexpectedFile {
            model_id: "alpha".to_owned(),
            path: "onnx/encoder_model.onnx".to_owned()
        })
    );
}

#[test]
fn rejects_unknown_manifest_field() {
    let result = registry_with(|v| v["models"][0]["extra"] = true.into());
    assert!(matches!(result, Err(WebManifestError::Parse { .. })));
}

#[test]
fn digest_round_trips_through_text() -> Result<(), crate::web_model::digest::Sha256ParseError> {
    let digest: Sha256Digest = DIGEST.parse()?;
    assert_eq!(digest.to_string(), DIGEST);
    assert!(!digest.is_zero());
    Ok(())
}

#[test]
fn digest_rejects_wrong_length() {
    assert!("abcd".parse::<Sha256Digest>().is_err());
}
