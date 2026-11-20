use super::*;
use crate::model::family::WhisperSize;
use serde_json::{Value, json};

const DIGEST: &str = "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21";
const ZERO_DIGEST: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";

fn files_json(
    roles: impl Fn(&ModelFileRole) -> bool,
    encoder: OnnxDtype,
    decoder: OnnxDtype,
) -> Value {
    ModelFileRole::ALL
        .iter()
        .filter(|role| roles(role))
        .zip(1u64..)
        .map(|(role, size)| {
            let mut file = json!({
                "path": role.path(encoder, decoder),
                "size_bytes": size * 100,
                "sha256": DIGEST,
            });
            if role.is_onnx() {
                file["converted"] = json!({
                    "graph": {"size_bytes": size * 10, "sha256": DIGEST},
                    "data": {"size_bytes": size * 90, "sha256": DIGEST},
                });
            }
            file
        })
        .collect()
}

fn plain_json() -> Value {
    files_json(|role| !role.is_onnx(), OnnxDtype::Fp32, OnnxDtype::Fp32)
}

fn variant_json(precision: &str, encoder: OnnxDtype, decoder: OnnxDtype, table: &str) -> Value {
    let mut variant = json!({
        "precision": precision,
        "encoder_dtype": encoder.to_string(),
        "decoder_dtype": decoder.to_string(),
        "files": files_json(|role| role.is_onnx(), encoder, decoder),
    });
    if table != "fp32" {
        variant["table"] = table.into();
    }
    variant
}

fn manifest_json(edit: impl FnOnce(&mut Value)) -> String {
    let mut value = json!({
        "schema": SUPPORTED_SCHEMA,
        "default": "alpha",
        "models": [
            {
                "id": "alpha", "repo": "owner/alpha", "revision": REVISION,
                "family": "base", "languages": "any",
                "description": "Alpha",
                "files": plain_json(),
                "variants": [
                    variant_json("fp16", OnnxDtype::Fp16, OnnxDtype::Q4, "fp16"),
                    variant_json("fp32", OnnxDtype::Fp32, OnnxDtype::Q4, "fp32"),
                ],
            },
            {
                "id": "beta", "repo": "owner/beta", "revision": REVISION,
                "family": "large-v3-turbo", "languages": "any",
                "description": "Beta",
                "files": plain_json(),
                "variants": [variant_json("fp16", OnnxDtype::Fp16, OnnxDtype::Q4, "fp16")],
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
fn variants_are_typed_and_the_full_precision_one_comes_first()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|_| {})?;
    let alpha = registry.get("alpha")?;
    let layout: Vec<(VariantKind, OnnxDtype, OnnxDtype, TableStorage)> = alpha
        .variants()
        .map(|spec| {
            (
                spec.variant,
                spec.encoder_dtype,
                spec.decoder_dtype,
                spec.table,
            )
        })
        .collect();
    assert_eq!(
        layout,
        [
            (
                VariantKind::Fp32,
                OnnxDtype::Fp32,
                OnnxDtype::Q4,
                TableStorage::Fp32
            ),
            (
                VariantKind::Fp16,
                OnnxDtype::Fp16,
                OnnxDtype::Q4,
                TableStorage::Fp16
            ),
        ]
    );
    assert_eq!(
        registry.get("beta")?.baseline().family,
        ModelFamily::Whisper(WhisperSize::LargeV3Turbo)
    );
    Ok(())
}

#[test]
fn english_only_follows_the_language() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|v| {
        v["models"][0]["languages"] = json!({"only": ["en"]});
    })?;
    assert!(registry.get("alpha")?.baseline().english_only());
    assert!(!registry.get("beta")?.baseline().english_only());
    Ok(())
}

#[test]
fn files_follow_role_order_and_the_paths_of_the_variant() -> Result<(), Box<dyn std::error::Error>>
{
    let registry = registry_with(|v| {
        if let Some(files) = v["models"][0]["variants"][1]["files"].as_array_mut() {
            files.reverse();
        }
    })?;
    let alpha = registry.get("alpha")?;
    let layout = |kind| -> Result<Vec<(ModelFileRole, String)>, Box<dyn std::error::Error>> {
        Ok(alpha
            .variant(kind)
            .ok_or("no variant")?
            .files
            .iter()
            .map(|file| (file.role, file.path.clone()))
            .collect())
    };
    let shared = [
        (ModelFileRole::Config, "config.json".to_owned()),
        (
            ModelFileRole::GenerationConfig,
            "generation_config.json".to_owned(),
        ),
        (
            ModelFileRole::PreprocessorConfig,
            "preprocessor_config.json".to_owned(),
        ),
        (ModelFileRole::Tokenizer, "tokenizer.json".to_owned()),
        (
            ModelFileRole::TokenizerConfig,
            "tokenizer_config.json".to_owned(),
        ),
    ];
    let with = |encoder: &str| {
        let mut files = shared.to_vec();
        files.push((ModelFileRole::Encoder, encoder.to_owned()));
        files.push((
            ModelFileRole::DecoderMerged,
            "onnx/decoder_model_merged_q4.onnx".to_owned(),
        ));
        files
    };
    assert_eq!(layout(VariantKind::Fp32)?, with("onnx/encoder_model.onnx"));
    assert_eq!(
        layout(VariantKind::Fp16)?,
        with("onnx/encoder_model_fp16.onnx")
    );
    Ok(())
}

#[test]
fn total_size_is_the_sum_of_every_file_of_the_variant() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|_| {})?;
    let alpha = registry.get("alpha")?;
    assert_eq!(alpha.baseline().total_size_bytes(), 1800);
    Ok(())
}

#[test]
fn urls_are_built_from_repo_revision_and_path() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|_| {})?;
    let alpha = registry.get("alpha")?.baseline();
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
fn only_a_model_without_a_full_precision_variant_requires_shader_f16()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|_| {})?;
    assert!(!registry.get("alpha")?.requires_shader_f16());
    assert!(registry.get("beta")?.requires_shader_f16());
    Ok(())
}

#[test]
fn webgpu_runs_half_precision_where_the_adapter_has_shader_f16()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|_| {})?;
    let alpha = registry.get("alpha")?;
    assert_eq!(
        alpha.webgpu_variant(ShaderF16::Supported).variant,
        VariantKind::Fp16
    );
    assert_eq!(
        alpha.webgpu_variant(ShaderF16::Missing).variant,
        VariantKind::Fp32
    );
    let beta = registry.get("beta")?;
    for support in [ShaderF16::Supported, ShaderF16::Missing] {
        assert_eq!(beta.webgpu_variant(support).variant, VariantKind::Fp16);
    }
    Ok(())
}

#[test]
fn the_cpu_runs_full_precision_and_only_that() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|_| {})?;
    assert_eq!(
        registry
            .get("alpha")?
            .cpu_variant()
            .map(|spec| spec.variant),
        Some(VariantKind::Fp32)
    );
    assert!(registry.get("beta")?.cpu_variant().is_none());
    assert!(registry.get("alpha")?.runs_on_cpu());
    assert!(!registry.get("beta")?.runs_on_cpu());
    Ok(())
}

#[test]
fn files_a_kept_sibling_stores_the_same_way_are_not_evicted()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|v| {
        v["models"][1]["repo"] = "owner/alpha".into();
    })?;
    let alpha = registry.get("alpha")?.baseline();
    let beta = registry.get("beta")?.baseline();
    let paths = |spec: &WebModelSpec, kept: &[&WebModelSpec]| -> Vec<String> {
        spec.files_not_loaded_by(kept)
            .into_iter()
            .map(|file| file.path.clone())
            .collect()
    };
    assert_eq!(paths(alpha, &[beta]), ["onnx/encoder_model.onnx"]);
    assert_eq!(paths(alpha, &[]).len(), alpha.files.len());
    Ok(())
}

#[test]
fn a_file_stored_differently_under_the_same_path_is_not_shared()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|v| {
        v["models"][0]["variants"][0]["files"][1]["converted"]["data"]["size_bytes"] = 1.into();
    })?;
    let alpha = registry.get("alpha")?;
    let full = alpha.variant(VariantKind::Fp32).ok_or("no full variant")?;
    let half = alpha.variant(VariantKind::Fp16).ok_or("no half variant")?;
    let evicted: Vec<ModelFileRole> = full
        .files_not_loaded_by(&[half])
        .into_iter()
        .map(|file| file.role)
        .collect();
    assert_eq!(
        evicted,
        [ModelFileRole::Encoder, ModelFileRole::DecoderMerged]
    );
    Ok(())
}

#[test]
fn files_of_a_model_in_another_repository_are_always_evicted()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|_| {})?;
    let alpha = registry.get("alpha")?.baseline();
    let beta = registry.get("beta")?.baseline();
    assert_eq!(alpha.files_not_loaded_by(&[beta]).len(), alpha.files.len());
    Ok(())
}

#[test]
fn siblings_share_a_repository_revision() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|v| {
        v["models"][1]["repo"] = "owner/alpha".into();
    })?;
    let ids = |id: &str| -> Result<Vec<&str>, Box<dyn std::error::Error>> {
        Ok(registry
            .siblings(registry.get(id)?)
            .map(|sibling| sibling.id.as_str())
            .collect())
    };
    assert_eq!(ids("alpha")?, ["beta"]);
    let separate = registry_with(|_| {})?;
    assert_eq!(separate.siblings(separate.get("alpha")?).count(), 0);
    Ok(())
}

#[test]
fn only_dtypes_the_webgpu_provider_runs_are_accepted() {
    let supported: Vec<OnnxDtype> = [
        OnnxDtype::Fp32,
        OnnxDtype::Fp16,
        OnnxDtype::Q8,
        OnnxDtype::Int8,
        OnnxDtype::Uint8,
        OnnxDtype::Q4,
        OnnxDtype::Bnb4,
        OnnxDtype::Q4F16,
    ]
    .into_iter()
    .filter(|dtype| dtype.runs_on_webgpu())
    .collect();
    assert_eq!(
        supported,
        [
            OnnxDtype::Fp32,
            OnnxDtype::Fp16,
            OnnxDtype::Q4,
            OnnxDtype::Q4F16
        ]
    );
}

#[test]
fn rejects_a_dtype_webgpu_cannot_run() {
    let result = registry_with(|v| {
        v["models"][0]["variants"][1] =
            variant_json("fp32", OnnxDtype::Fp32, OnnxDtype::Q8, "fp32");
    });
    assert_eq!(
        result.err(),
        Some(WebManifestError::UnsupportedDtype {
            model_id: "alpha".to_owned(),
            dtype: OnnxDtype::Q8
        })
    );
}

#[test]
fn rejects_a_full_precision_variant_that_needs_shader_f16() {
    let edits: [fn(&mut Value); 2] = [
        |v: &mut Value| {
            v["models"][0]["variants"][1] =
                variant_json("fp32", OnnxDtype::Fp16, OnnxDtype::Q4, "fp32");
        },
        |v: &mut Value| {
            v["models"][0]["variants"][1] =
                variant_json("fp32", OnnxDtype::Fp32, OnnxDtype::Q4, "fp16");
        },
    ];
    for edit in edits {
        assert_eq!(
            registry_with(edit).err(),
            Some(WebManifestError::InconsistentVariant {
                model_id: "alpha".to_owned(),
                kind: VariantKind::Fp32
            })
        );
    }
}

#[test]
fn rejects_a_model_without_variants() {
    let result = registry_with(|v| v["models"][0]["variants"] = json!([]));
    assert_eq!(
        result.err(),
        Some(WebManifestError::NoVariants {
            model_id: "alpha".to_owned()
        })
    );
}

#[test]
fn rejects_a_variant_listed_twice() {
    let result = registry_with(|v| {
        v["models"][0]["variants"][1] =
            variant_json("fp16", OnnxDtype::Fp16, OnnxDtype::Q4, "fp16");
    });
    assert_eq!(
        result.err(),
        Some(WebManifestError::DuplicateVariant {
            model_id: "alpha".to_owned(),
            kind: VariantKind::Fp16
        })
    );
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
    let result = registry_with(|v| v["schema"] = 1.into());
    assert_eq!(
        result.err(),
        Some(WebManifestError::UnsupportedSchema {
            found: 1,
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
        if let Some(files) = v["models"][0]["variants"][1]["files"].as_array_mut() {
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
    let result = registry_with(|v| v["models"][0]["variants"][0]["encoder_dtype"] = "fp32".into());
    assert_eq!(
        result.err(),
        Some(WebManifestError::UnexpectedFile {
            model_id: "alpha".to_owned(),
            path: "onnx/encoder_model_fp16.onnx".to_owned()
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

#[test]
fn embedded_manifest_parses() -> Result<(), WebManifestError> {
    let registry = WebModelRegistry::embedded()?;
    let ids: Vec<&str> = registry.models().iter().map(|m| m.id.as_str()).collect();
    assert_eq!(
        ids,
        [
            "tiny",
            "tiny.en",
            "base",
            "base-q4",
            "base.en",
            "base.en-q4",
            "small",
            "small-q4",
            "small.en",
            "small.en-q4",
            "large-v3-turbo",
            "large-v3-turbo-q4f16"
        ]
    );
    assert_eq!(registry.default_model().id, "base");
    Ok(())
}

#[test]
fn embedded_manifest_pins_every_variant_to_a_complete_file_set() -> Result<(), WebManifestError> {
    let registry = WebModelRegistry::embedded()?;
    for model in registry.models() {
        for spec in model.variants() {
            assert_eq!(
                spec.files.len(),
                ModelFileRole::ALL.len(),
                "{} {}",
                model.id,
                spec.variant
            );
            assert!(spec.total_size_bytes() > 0, "{}", model.id);
        }
    }
    Ok(())
}

#[test]
fn only_the_dot_en_models_are_english_only() -> Result<(), WebManifestError> {
    let registry = WebModelRegistry::embedded()?;
    for model in registry.models() {
        assert_eq!(
            model.baseline().english_only(),
            matches!(
                model.id.as_str(),
                "tiny.en" | "base.en" | "base.en-q4" | "small.en" | "small.en-q4"
            ),
            "{}",
            model.id
        );
    }
    Ok(())
}

#[test]
fn embedded_models_have_the_names_the_picker_showed() -> Result<(), WebManifestError> {
    let registry = WebModelRegistry::embedded()?;
    let names: Vec<String> = registry
        .models()
        .iter()
        .map(|m| m.baseline().display_name())
        .collect();
    assert_eq!(
        names,
        [
            "Whisper Tiny",
            "Whisper Tiny (English)",
            "Whisper Base",
            "Whisper Base",
            "Whisper Base (English)",
            "Whisper Base (English)",
            "Whisper Small",
            "Whisper Small",
            "Whisper Small (English)",
            "Whisper Small (English)",
            "Whisper Large v3 Turbo",
            "Whisper Large v3 Turbo",
        ]
    );
    Ok(())
}

#[test]
fn only_the_embedded_turbo_models_require_shader_f16() -> Result<(), WebManifestError> {
    let registry = WebModelRegistry::embedded()?;
    let needing: Vec<&str> = registry
        .models()
        .iter()
        .filter(|model| model.requires_shader_f16())
        .map(|model| model.id.as_str())
        .collect();
    assert_eq!(needing, ["large-v3-turbo", "large-v3-turbo-q4f16"]);
    Ok(())
}

#[test]
fn only_the_embedded_half_precision_turbo_cannot_run_on_the_cpu() -> Result<(), WebManifestError> {
    let registry = WebModelRegistry::embedded()?;
    let gpu_only: Vec<&str> = registry
        .models()
        .iter()
        .filter(|model| !model.runs_on_cpu())
        .map(|model| model.id.as_str())
        .collect();
    assert_eq!(gpu_only, ["large-v3-turbo"]);
    Ok(())
}

#[test]
fn every_embedded_model_that_runs_on_the_cpu_also_has_a_half_precision_variant()
-> Result<(), WebManifestError> {
    let registry = WebModelRegistry::embedded()?;
    for model in registry.models() {
        let cpu = model.cpu_variant().map(|spec| spec.variant);
        let half = model.variant(VariantKind::Fp16).map(|spec| spec.variant);
        assert_eq!(half, Some(VariantKind::Fp16), "{}", model.id);
        if model.variant(VariantKind::Fp32).is_some() {
            assert_eq!(cpu, Some(VariantKind::Fp32), "{}", model.id);
        }
    }
    Ok(())
}

#[test]
fn the_embedded_half_precision_variants_are_smaller_than_the_full_precision_ones()
-> Result<(), WebManifestError> {
    let registry = WebModelRegistry::embedded()?;
    for model in registry.models() {
        let (Some(half), Some(full)) = (
            model.variant(VariantKind::Fp16),
            model.variant(VariantKind::Fp32),
        ) else {
            continue;
        };
        let stored = |spec: &WebModelSpec| -> u64 {
            spec.files
                .iter()
                .flat_map(WebModelFile::cached_files)
                .map(|file| file.size_bytes)
                .sum()
        };
        assert!(stored(half) < stored(full), "{}", model.id);
    }
    Ok(())
}

#[test]
fn embedded_quantized_models_are_smaller_than_the_default_of_their_family()
-> Result<(), WebManifestError> {
    let registry = WebModelRegistry::embedded()?;
    let is_quantized = |model: &WebModel| {
        matches!(
            model.baseline().encoder_dtype,
            OnnxDtype::Q4 | OnnxDtype::Q4F16
        )
    };
    for variant in registry.models().iter().filter(|model| is_quantized(model)) {
        let original = registry
            .models()
            .iter()
            .find(|model| {
                model.baseline().family == variant.baseline().family && !is_quantized(model)
            })
            .ok_or_else(|| WebManifestError::UnknownDefaultModel {
                id: variant.id.clone(),
            })?;
        assert!(
            variant.baseline().total_size_bytes() < original.baseline().total_size_bytes(),
            "{}",
            variant.id
        );
    }
    Ok(())
}

#[test]
fn the_q4_model_shares_everything_but_the_encoder_with_the_original()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = WebModelRegistry::embedded()?;
    let base = registry.get("base")?;
    let q4 = registry.get("base-q4")?;
    for kind in [VariantKind::Fp16, VariantKind::Fp32] {
        let (Some(base), Some(q4)) = (base.variant(kind), q4.variant(kind)) else {
            return Err("a variant is missing".into());
        };
        let evicted: Vec<ModelFileRole> = q4
            .files_not_loaded_by(&[base])
            .into_iter()
            .map(|file| file.role)
            .collect();
        assert_eq!(evicted, [ModelFileRole::Encoder], "{kind}");
    }
    Ok(())
}

#[test]
fn an_onnx_file_without_conversion_is_rejected() {
    let result = registry_with(|v| {
        if let Some(file) = v["models"][0]["variants"][1]["files"]
            .as_array_mut()
            .and_then(|f| f.last_mut())
        {
            file.as_object_mut().map(|file| file.remove("converted"));
        }
    });
    assert!(matches!(
        result.err(),
        Some(WebManifestError::MissingConversion { .. })
    ));
}

#[test]
fn a_conversion_on_a_plain_file_is_rejected() {
    let result = registry_with(|v| {
        v["models"][0]["files"][0]["converted"] = json!({
            "graph": {"size_bytes": 1, "sha256": DIGEST},
            "data": {"size_bytes": 1, "sha256": DIGEST},
        });
    });
    assert!(matches!(
        result.err(),
        Some(WebManifestError::UnexpectedConversion { .. })
    ));
}

#[test]
fn a_converted_file_is_cached_as_its_graph_and_its_weights()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|_| {})?;
    let spec = registry.get("alpha")?.baseline();
    let decoder = spec.files.last().ok_or("no files")?;
    let cached = decoder.cached_files();
    assert_eq!(
        cached
            .iter()
            .map(|file| (file.path.as_str(), file.size_bytes))
            .collect::<Vec<_>>(),
        [
            ("onnx/decoder_model_merged_q4.onnx", 20),
            ("onnx/decoder_model_merged_q4.onnx_data", 180),
        ]
    );
    let config = spec.files.first().ok_or("no files")?;
    assert_eq!(config.cached_files().len(), 1);
    Ok(())
}
