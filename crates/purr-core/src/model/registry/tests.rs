use super::*;
use crate::model::digest::{Sha256Hasher, Sha256ParseError};
use std::str::FromStr;

const DIGEST_A: &str = "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21";
const DIGEST_B: &str = "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002";
const REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";

fn manifest_json(edit: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut value = serde_json::json!({
        "schema": 1,
        "preferred": ["alpha", "beta.en-tdrz"],
        "sources": [
            {"id": "main", "repo": "owner/repo", "revision": REVISION},
            {"id": "extra", "repo": "other/repo", "revision": REVISION},
        ],
        "models": [
            {
                "id": "alpha", "source": "main", "filename": "ggml-alpha.bin",
                "family": "tiny", "language": "multilingual", "quantization": "fp16",
                "features": [], "size_bytes": 10, "sha256": DIGEST_A,
                "description": "Alpha"
            },
            {
                "id": "beta.en-tdrz", "source": "extra", "filename": "ggml-beta.en-tdrz.bin",
                "family": "small", "language": "english-only", "quantization": "q5_1",
                "features": ["tdrz"], "size_bytes": 20, "sha256": DIGEST_B,
                "description": "Beta"
            },
        ],
    });
    edit(&mut value);
    value.to_string()
}

fn registry_with(edit: impl FnOnce(&mut serde_json::Value)) -> Result<Registry, ManifestError> {
    Registry::from_json(&manifest_json(edit))
}

#[test]
fn embedded_manifest_parses() -> Result<(), ManifestError> {
    let registry = Registry::embedded()?;
    assert_eq!(registry.models().len(), 34);
    assert_eq!(registry.default_model().id, "base");
    Ok(())
}

#[test]
fn embedded_manifest_entries_are_complete() -> Result<(), ManifestError> {
    let registry = Registry::embedded()?;
    let mut ids = HashSet::new();
    let mut filenames = HashSet::new();
    for model in registry.models() {
        assert!(ids.insert(model.id.as_str()), "duplicate id {}", model.id);
        assert!(
            filenames.insert(model.filename.as_str()),
            "duplicate filename {}",
            model.filename
        );
        assert!(model.size_bytes > 0, "{} has no size", model.id);
        let hex = model.sha256.to_string();
        assert_eq!(hex.len(), 64, "{} digest length", model.id);
        assert!(
            hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
            "{} digest is not lowercase hex",
            model.id
        );
        assert!(
            !model.sha256.is_zero(),
            "{} has a placeholder digest",
            model.id
        );
    }
    Ok(())
}

#[test]
fn embedded_preferred_models_resolve() -> Result<(), ManifestError> {
    let registry = Registry::embedded()?;
    let preferred: Vec<&str> = registry.preferred().map(|m| m.id.as_str()).collect();
    assert_eq!(
        preferred,
        ["base", "base.en", "small", "small.en", "tiny", "tiny.en"]
    );
    Ok(())
}

#[test]
fn url_is_built_from_source_revision_and_filename() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|_| {})?;
    assert_eq!(
        registry.get("alpha")?.url(),
        format!("https://huggingface.co/owner/repo/resolve/{REVISION}/ggml-alpha.bin")
    );
    assert_eq!(
        registry.get("beta.en-tdrz")?.url(),
        format!("https://huggingface.co/other/repo/resolve/{REVISION}/ggml-beta.en-tdrz.bin")
    );
    Ok(())
}

#[test]
fn embedded_tdrz_model_resolves_to_its_own_repository() -> Result<(), Box<dyn std::error::Error>> {
    let registry = Registry::embedded()?;
    let tdrz = registry.get("small.en-tdrz")?;
    assert_eq!(tdrz.source.repo, "akashmjn/tinydiarize-whisper.cpp");
    assert!(tdrz.has_feature(ModelFeature::Tdrz));
    Ok(())
}

#[test]
fn lookup_of_unknown_id_is_a_typed_error() -> Result<(), ManifestError> {
    let registry = registry_with(|_| {})?;
    let error = registry.get("gamma").err();
    assert_eq!(error.map(|e| e.model_name), Some("gamma".to_owned()));
    Ok(())
}

#[test]
fn attributes_are_typed() -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry_with(|_| {})?;
    let beta = registry.get("beta.en-tdrz")?;
    assert_eq!(beta.family, ModelFamily::Small);
    assert_eq!(beta.language, ModelLanguage::EnglishOnly);
    assert_eq!(beta.quantization, Quantization::Q5_1);
    assert_eq!(beta.features, [ModelFeature::Tdrz]);
    Ok(())
}

#[test]
fn groups_partition_models_by_family_in_ascending_order() -> Result<(), ManifestError> {
    let registry = Registry::embedded()?;
    let groups = registry.groups();
    let families: Vec<ModelFamily> = groups.iter().map(|g| g.family).collect();
    let mut sorted = families.clone();
    sorted.sort();
    assert_eq!(families, sorted);
    assert_eq!(
        groups.iter().map(|g| g.models.len()).sum::<usize>(),
        registry.models().len()
    );
    assert!(
        groups
            .iter()
            .all(|g| g.models.iter().all(|m| m.family == g.family))
    );
    Ok(())
}

#[test]
fn rejects_unsupported_schema() {
    let result = registry_with(|v| v["schema"] = 2.into());
    assert_eq!(
        result.err(),
        Some(ManifestError::UnsupportedSchema {
            found: 2,
            supported: SUPPORTED_SCHEMA
        })
    );
}

#[test]
fn rejects_duplicate_model_id() {
    let result = registry_with(|v| v["models"][1]["id"] = "alpha".into());
    assert_eq!(
        result.err(),
        Some(ManifestError::DuplicateModelId {
            id: "alpha".to_owned()
        })
    );
}

#[test]
fn rejects_duplicate_filename() {
    let result = registry_with(|v| v["models"][1]["filename"] = "ggml-alpha.bin".into());
    assert_eq!(
        result.err(),
        Some(ManifestError::DuplicateFilename {
            filename: "ggml-alpha.bin".to_owned()
        })
    );
}

#[test]
fn rejects_duplicate_source_id() {
    let result = registry_with(|v| v["sources"][1]["id"] = "main".into());
    assert_eq!(
        result.err(),
        Some(ManifestError::DuplicateSourceId {
            id: "main".to_owned()
        })
    );
}

#[test]
fn rejects_unknown_source() {
    let result = registry_with(|v| v["models"][0]["source"] = "nowhere".into());
    assert_eq!(
        result.err(),
        Some(ManifestError::UnknownSource {
            model_id: "alpha".to_owned(),
            source_id: "nowhere".to_owned()
        })
    );
}

#[test]
fn rejects_invalid_revision() {
    let result = registry_with(|v| v["sources"][0]["revision"] = "main".into());
    assert_eq!(
        result.err(),
        Some(ManifestError::InvalidRevision {
            source_id: "main".to_owned(),
            revision: "main".to_owned()
        })
    );
}

#[test]
fn rejects_zero_size() {
    let result = registry_with(|v| v["models"][0]["size_bytes"] = 0.into());
    assert_eq!(
        result.err(),
        Some(ManifestError::ZeroSize {
            model_id: "alpha".to_owned()
        })
    );
}

#[test]
fn rejects_placeholder_digest() {
    let zero = "0".repeat(64);
    let result = registry_with(|v| v["models"][0]["sha256"] = zero.into());
    assert_eq!(
        result.err(),
        Some(ManifestError::MissingDigest {
            model_id: "alpha".to_owned()
        })
    );
}

#[test]
fn rejects_unknown_preferred_model() {
    let result = registry_with(|v| v["preferred"] = serde_json::json!(["alpha", "gamma"]));
    assert_eq!(
        result.err(),
        Some(ManifestError::UnknownPreferredModel {
            id: "gamma".to_owned()
        })
    );
}

#[test]
fn rejects_empty_preferred_list() {
    let result = registry_with(|v| v["preferred"] = serde_json::json!([]));
    assert_eq!(result.err(), Some(ManifestError::NoPreferredModel));
}

#[test]
fn rejects_malformed_digest_and_unknown_fields() {
    let short = registry_with(|v| v["models"][0]["sha256"] = "abcd".into());
    assert!(matches!(short, Err(ManifestError::Parse { .. })));
    let extra = registry_with(|v| v["models"][0]["colour"] = "red".into());
    assert!(matches!(extra, Err(ManifestError::Parse { .. })));
}

#[test]
fn digest_round_trips_through_text() -> Result<(), Sha256ParseError> {
    let digest = Sha256Digest::from_str(DIGEST_A)?;
    assert_eq!(digest.to_string(), DIGEST_A);
    assert_eq!(Sha256Digest::from_str(&DIGEST_A.to_uppercase())?, digest);
    assert!(Sha256Digest::from_str(&DIGEST_A[1..]).is_err());
    Ok(())
}

#[test]
fn hasher_matches_known_digest() -> Result<(), Sha256ParseError> {
    let mut hasher = Sha256Hasher::new();
    hasher.update(b"a");
    hasher.update(b"bc");
    assert_eq!(
        hasher.finalize(),
        Sha256Digest::from_str("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")?
    );
    Ok(())
}
