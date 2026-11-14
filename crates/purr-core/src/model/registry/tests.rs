use super::*;
use crate::model::digest::{Sha256Hasher, Sha256ParseError};
use purr_common::language::support::LanguageSupport;
use purr_common::model::profile::{DetectionReport, HintEffect, InputLimit, Speakers, Timing};
use purr_common::model::{
    engine::EngineId,
    family::{ModelFamily, ParakeetVariant, Qwen3AsrSize, WhisperSize},
    format::ModelFormat,
    quantization::Quantization,
};
use std::str::FromStr;

const DIGEST_A: &str = "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21";
const DIGEST_B: &str = "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002";
const DIGEST_C: &str = "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987";
const REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";

fn manifest_json(edit: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut value = serde_json::json!({
        "schema": 3,
        "preferred": ["alpha", "beta.en-tdrz"],
        "sources": [
            {"id": "main", "repo": "owner/repo", "revision": REVISION},
            {"id": "extra", "repo": "other/repo", "revision": REVISION},
        ],
        "profiles": {
            "plain": {
                "languages": "any", "language_hint": "steers", "detection": "reported",
                "timing": "word", "word_scores": true, "speakers": "none", "input": "unbounded"
            },
            "english-turns": {
                "languages": {"only": ["en"]}, "language_hint": "steers", "detection": "reported",
                "timing": "word", "word_scores": true, "speakers": "turn-markers",
                "input": "unbounded"
            },
        },
        "models": [
            {
                "id": "alpha", "engine": "whisper-cpp", "format": "ggml-bin",
                "source": "main", "filename": "ggml-alpha.bin",
                "family": "tiny", "profile": "plain", "quantization": "f16",
                "size_bytes": 10, "sha256": DIGEST_A,
                "description": "Alpha"
            },
            {
                "id": "beta.en-tdrz", "engine": "whisper-cpp", "format": "ggml-bin",
                "source": "extra", "filename": "ggml-beta.en-tdrz.bin",
                "family": "small", "profile": "english-turns", "quantization": "q5_1",
                "size_bytes": 20, "sha256": DIGEST_B,
                "description": "Beta"
            },
        ],
        "vad": {
            "id": "gamma-vad", "source": "extra", "filename": "ggml-gamma-vad.bin",
            "size_bytes": 30, "sha256": DIGEST_C
        },
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
    assert_eq!(registry.models().len(), 43);
    assert_eq!(registry.vad().id, "silero-v6.2.0");
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
        assert!(
            model.engine.supports(model.format),
            "{} is run by an engine that does not load its format",
            model.id
        );
        assert!(model.file.size_bytes > 0, "{} has no size", model.id);
        let hex = model.file.sha256.to_string();
        assert_eq!(hex.len(), 64, "{} digest length", model.id);
        assert!(
            hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
            "{} digest is not lowercase hex",
            model.id
        );
        assert!(
            !model.file.sha256.is_zero(),
            "{} has a placeholder digest",
            model.id
        );
    }
    Ok(())
}

#[test]
fn embedded_voice_activity_model_is_pinned_to_the_published_file() -> Result<(), ManifestError> {
    let registry = Registry::embedded()?;
    let vad = registry.vad();
    assert_eq!(vad.filename, "ggml-silero-v6.2.0.bin");
    assert_eq!(vad.source.repo, "ggml-org/whisper-vad");
    assert_eq!(
        vad.source.revision,
        "9ffd54a1e1ee413ddf265af9913beaf518d1639b"
    );
    assert_eq!(vad.file.size_bytes, 885_098);
    assert_eq!(vad.file.sha256.to_string(), DIGEST_C);
    assert_eq!(
        vad.file.url,
        "https://huggingface.co/ggml-org/whisper-vad/resolve/9ffd54a1e1ee413ddf265af9913beaf518d1639b/ggml-silero-v6.2.0.bin"
    );
    Ok(())
}

#[test]
fn embedded_voice_activity_model_is_not_a_whisper_model() -> Result<(), ManifestError> {
    let registry = Registry::embedded()?;
    let vad = registry.vad();
    assert!(registry.models().iter().all(|model| model.id != vad.id));
    assert!(
        registry
            .models()
            .iter()
            .all(|model| model.filename != vad.filename)
    );
    assert!(
        registry
            .groups()
            .iter()
            .all(|group| group.models.iter().all(|model| model.id != vad.id))
    );
    assert_eq!(
        registry.get(&vad.id).err().map(|e| e.model_name),
        Some(vad.id.clone())
    );
    Ok(())
}

#[test]
fn voice_activity_model_resolves_its_source_and_url() -> Result<(), ManifestError> {
    let registry = registry_with(|_| {})?;
    let vad = registry.vad();
    assert_eq!(vad.id, "gamma-vad");
    assert_eq!(vad.source.repo, "other/repo");
    assert_eq!(vad.file.name, "gamma-vad");
    assert_eq!(vad.file.size_bytes, 30);
    assert_eq!(
        vad.file.url,
        format!("https://huggingface.co/other/repo/resolve/{REVISION}/ggml-gamma-vad.bin")
    );
    assert_eq!(registry.models().len(), 2);
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
    assert_eq!(tdrz.profile.speakers, Speakers::TurnMarkers);
    Ok(())
}

#[test]
fn embedded_parakeet_models_are_gguf_files_for_transcribe_cpp()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = Registry::embedded()?;
    let q4 = registry.get("parakeet-tdt-0.6b-v3-q4_k_m")?;
    assert_eq!(q4.engine, EngineId::TranscribeCpp);
    assert_eq!(q4.format, ModelFormat::Gguf);
    assert_eq!(q4.family, ModelFamily::Parakeet(ParakeetVariant::TdtV3));
    assert_eq!(q4.quantization, Quantization::Q4KM);
    assert_eq!(q4.file.size_bytes, 485_425_504);
    assert_eq!(
        q4.url(),
        "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf/resolve/d8d8c32cb2017c57e5307767343e97870f8a7f53/parakeet-tdt-0.6b-v3-Q4_K_M.gguf"
    );
    let engines: Vec<EngineId> = registry
        .groups()
        .into_iter()
        .filter(|group| matches!(group.family, ModelFamily::Parakeet(_)))
        .flat_map(|group| group.models)
        .map(|model| model.engine)
        .collect();
    assert_eq!(engines, [EngineId::TranscribeCpp; 3]);
    Ok(())
}

#[test]
fn embedded_parakeet_profile_says_it_finds_the_language_without_telling()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = Registry::embedded()?;
    let profile = &registry.get("parakeet-tdt-0.6b-v3")?.profile;
    assert_eq!(profile.language_hint, HintEffect::Ignored);
    assert_eq!(profile.detection, DetectionReport::Silent);
    assert_eq!(profile.timing, Timing::Word);
    assert_eq!(profile.input, InputLimit::Unbounded);
    assert!(profile.languages.supports(&"pt-PT".parse()?));
    assert!(!profile.languages.supports(&"ja".parse()?));
    assert!(profile.can_detect());
    Ok(())
}

#[test]
fn embedded_qwen3_asr_models_are_gguf_files_of_two_sizes_pinned_to_their_repositories()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = Registry::embedded()?;
    let small = registry.get("qwen3-asr-0.6b-q8_0")?;
    assert_eq!(small.engine, EngineId::TranscribeCpp);
    assert_eq!(small.format, ModelFormat::Gguf);
    assert_eq!(small.family, ModelFamily::Qwen3Asr(Qwen3AsrSize::B0_6));
    assert_eq!(small.quantization, Quantization::Q8_0);
    assert_eq!(small.file.size_bytes, 850_423_456);
    assert_eq!(
        small.url(),
        "https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/7e7c69a577fa2725dfedbe34b5e999336e30a19b/Qwen3-ASR-0.6B-Q8_0.gguf"
    );
    let large = registry.get("qwen3-asr-1.7b-q4_k_m")?;
    assert_eq!(large.family, ModelFamily::Qwen3Asr(Qwen3AsrSize::B1_7));
    assert_eq!(
        large.url(),
        "https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/0e0a596e3959c3f3f2ab7d443a5b6f0f4a4bba6f/Qwen3-ASR-1.7B-Q4_K_M.gguf"
    );
    let per_family: Vec<(ModelFamily, usize)> = registry
        .groups()
        .into_iter()
        .filter(|group| matches!(group.family, ModelFamily::Qwen3Asr(_)))
        .map(|group| (group.family, group.models.len()))
        .collect();
    assert_eq!(
        per_family,
        [
            (ModelFamily::Qwen3Asr(Qwen3AsrSize::B0_6), 3),
            (ModelFamily::Qwen3Asr(Qwen3AsrSize::B1_7), 3)
        ]
    );
    Ok(())
}

#[test]
fn embedded_qwen3_asr_profile_says_the_choice_steers_and_the_text_has_no_times()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = Registry::embedded()?;
    let profile = &registry.get("qwen3-asr-1.7b")?.profile;
    assert_eq!(profile.language_hint, HintEffect::Steers);
    assert_eq!(profile.detection, DetectionReport::Reported);
    assert_eq!(profile.timing, Timing::None);
    assert!(!profile.word_scores);
    assert!(matches!(profile.input, InputLimit::UpTo(_)));
    assert!(profile.languages.supports(&"fr".parse()?));
    assert!(profile.languages.supports(&"yue".parse()?));
    assert!(!profile.languages.supports(&"he".parse()?));
    assert!(profile.can_detect());
    assert_eq!(
        registry.get("qwen3-asr-0.6b")?.profile,
        registry.get("qwen3-asr-1.7b")?.profile
    );
    Ok(())
}

#[test]
fn no_embedded_preferred_model_needs_the_transcribe_engine() -> Result<(), ManifestError> {
    let registry = Registry::embedded()?;
    assert!(
        registry
            .preferred()
            .all(|model| model.engine == EngineId::WhisperCpp)
    );
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
    assert_eq!(beta.engine, EngineId::WhisperCpp);
    assert_eq!(beta.format, ModelFormat::GgmlBin);
    assert_eq!(beta.family, ModelFamily::Whisper(WhisperSize::Small));
    assert_eq!(beta.profile.languages, LanguageSupport::english_only());
    assert_eq!(beta.profile.speakers, Speakers::TurnMarkers);
    assert_eq!(beta.quantization, Quantization::Q5_1);
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
    let result = registry_with(|v| v["schema"] = 1.into());
    assert_eq!(
        result.err(),
        Some(ManifestError::UnsupportedSchema {
            found: 1,
            supported: SUPPORTED_SCHEMA
        })
    );
}

#[test]
fn rejects_a_format_the_engine_does_not_load() {
    let result = registry_with(|v| v["models"][1]["format"] = "gguf".into());
    assert_eq!(
        result.err(),
        Some(ManifestError::UnsupportedFormat {
            model_id: "beta.en-tdrz".to_owned(),
            engine: EngineId::WhisperCpp,
            format: ModelFormat::Gguf,
        })
    );
}

#[test]
fn rejects_an_unknown_engine_and_a_model_without_one() {
    let unknown = registry_with(|v| v["models"][0]["engine"] = "vosk".into());
    assert!(matches!(unknown, Err(ManifestError::Parse { .. })));
    let missing = registry_with(|v| {
        if let Some(model) = v["models"][0].as_object_mut() {
            model.remove("engine");
        }
    });
    assert!(matches!(missing, Err(ManifestError::Parse { .. })));
}

#[test]
fn rejects_a_profile_the_manifest_does_not_define() {
    let result = registry_with(|v| v["models"][0]["profile"] = "vosk".into());
    assert_eq!(
        result.err(),
        Some(ManifestError::UnknownProfile {
            model_id: "alpha".to_owned(),
            profile: "vosk".to_owned(),
        })
    );
}

#[test]
fn rejects_a_profile_with_a_field_it_does_not_know() {
    let result = registry_with(|v| v["profiles"]["plain"]["translate"] = true.into());
    assert!(matches!(result, Err(ManifestError::Parse { .. })));
}

#[test]
fn rejects_the_fields_the_profile_replaced() {
    let language = registry_with(|v| v["models"][0]["language"] = "multilingual".into());
    assert!(matches!(language, Err(ManifestError::Parse { .. })));
    let features = registry_with(|v| v["models"][0]["features"] = serde_json::json!([]));
    assert!(matches!(features, Err(ManifestError::Parse { .. })));
}

#[test]
fn every_embedded_model_is_described_as_whisper_cpp_reports_it() -> Result<(), ManifestError> {
    let registry = Registry::embedded()?;
    for model in registry
        .models()
        .iter()
        .filter(|model| matches!(model.family, ModelFamily::Whisper(_)))
    {
        let english = model.id.starts_with("tiny.en")
            || model.id.starts_with("base.en")
            || model.id.starts_with("small.en")
            || model.id.starts_with("medium.en");
        let turns = model.id == "small.en-tdrz";
        let descriptor = model.descriptor();
        assert_eq!(descriptor.engine, EngineId::WhisperCpp, "{}", model.id);
        assert_eq!(descriptor.format, ModelFormat::GgmlBin, "{}", model.id);
        assert_eq!(
            descriptor.weights,
            Weights::Quantized(model.quantization),
            "{}",
            model.id
        );
        assert_eq!(
            descriptor.profile,
            ModelProfile {
                languages: if english {
                    LanguageSupport::english_only()
                } else {
                    LanguageSupport::Any
                },
                language_hint: HintEffect::Steers,
                detection: DetectionReport::Reported,
                timing: Timing::Word,
                word_scores: true,
                speakers: if turns {
                    Speakers::TurnMarkers
                } else {
                    Speakers::None
                },
                input: InputLimit::Unbounded,
            },
            "{}",
            model.id
        );
    }
    Ok(())
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
fn rejects_voice_activity_model_from_unknown_source() {
    let result = registry_with(|v| v["vad"]["source"] = "nowhere".into());
    assert_eq!(
        result.err(),
        Some(ManifestError::UnknownSource {
            model_id: "gamma-vad".to_owned(),
            source_id: "nowhere".to_owned()
        })
    );
}

#[test]
fn rejects_voice_activity_model_without_size_or_digest() {
    let no_size = registry_with(|v| v["vad"]["size_bytes"] = 0.into());
    assert_eq!(
        no_size.err(),
        Some(ManifestError::ZeroSize {
            model_id: "gamma-vad".to_owned()
        })
    );
    let zero = "0".repeat(64);
    let no_digest = registry_with(|v| v["vad"]["sha256"] = zero.into());
    assert_eq!(
        no_digest.err(),
        Some(ManifestError::MissingDigest {
            model_id: "gamma-vad".to_owned()
        })
    );
}

#[test]
fn rejects_voice_activity_model_that_collides_with_a_model() {
    let same_id = registry_with(|v| v["vad"]["id"] = "alpha".into());
    assert_eq!(
        same_id.err(),
        Some(ManifestError::DuplicateModelId {
            id: "alpha".to_owned()
        })
    );
    let same_file = registry_with(|v| v["vad"]["filename"] = "ggml-alpha.bin".into());
    assert_eq!(
        same_file.err(),
        Some(ManifestError::DuplicateFilename {
            filename: "ggml-alpha.bin".to_owned()
        })
    );
}

#[test]
fn rejects_manifest_without_voice_activity_model() {
    let result = registry_with(|v| {
        if let Some(object) = v.as_object_mut() {
            object.remove("vad");
        }
    });
    assert!(matches!(result, Err(ManifestError::Parse { .. })));
}

#[test]
fn rejects_malformed_voice_activity_model() {
    let short = registry_with(|v| v["vad"]["sha256"] = "abcd".into());
    assert!(matches!(short, Err(ManifestError::Parse { .. })));
    let extra = registry_with(|v| v["vad"]["family"] = "tiny".into());
    assert!(matches!(extra, Err(ManifestError::Parse { .. })));
    let no_file = registry_with(|v| {
        if let Some(vad) = v["vad"].as_object_mut() {
            vad.remove("filename");
        }
    });
    assert!(matches!(no_file, Err(ManifestError::Parse { .. })));
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
