//! Models for the tests of the catalog, built the way the platforms describe them.

use purr_common::{
    language::support::LanguageSupport,
    model::{
        descriptor::{ModelDescriptor, Weights},
        engine::EngineId,
        family::{ModelFamily, ParakeetVariant, Qwen3AsrSize, WhisperSize},
        format::ModelFormat,
        profile::{DetectionReport, HintEffect, InputLimit, ModelProfile, Speakers, Timing},
    },
    platform::ModelInfo,
    web_model::attributes::OnnxDtype,
};

/// What whisper.cpp reports of a Whisper model.
pub fn whisper_profile(english: bool, turns: bool) -> ModelProfile {
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
    }
}

/// What transcribe.cpp reports of Parakeet TDT v3: 25 languages, found by the model itself and
/// never named in the output.
pub fn parakeet_profile() -> ModelProfile {
    ModelProfile {
        languages: LanguageSupport::Only(
            [
                "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv",
                "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
            ]
            .iter()
            .map(|tag| tag.parse().unwrap())
            .collect(),
        ),
        language_hint: HintEffect::Ignored,
        detection: DetectionReport::Silent,
        timing: Timing::Token,
        word_scores: true,
        speakers: Speakers::None,
        input: InputLimit::Unbounded,
    }
}

/// A Parakeet TDT v3 model file of the catalog stored as `quantization`.
pub fn parakeet_info(id: &str, quantization: &str, size: u64, installed: bool) -> ModelInfo {
    info_of(
        id,
        size,
        installed,
        ModelDescriptor {
            engine: EngineId::TranscribeCpp,
            family: ModelFamily::Parakeet(ParakeetVariant::TdtV3),
            format: ModelFormat::Gguf,
            weights: weights(Some(quantization)),
            profile: parakeet_profile(),
        },
    )
}

/// What transcribe.cpp reports of Qwen3-ASR: 30 languages that the choice steers, a detection
/// the model names, and text without times.
pub fn qwen3_profile() -> ModelProfile {
    ModelProfile {
        languages: LanguageSupport::Only(
            [
                "zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id", "it", "ko", "ru", "th",
                "vi", "ja", "tr", "hi", "ms", "nl", "sv", "da", "fi", "pl", "cs", "fil", "fa",
                "el", "ro", "hu", "mk",
            ]
            .iter()
            .map(|tag| tag.parse().unwrap())
            .collect(),
        ),
        language_hint: HintEffect::Steers,
        detection: DetectionReport::Reported,
        timing: Timing::None,
        word_scores: false,
        speakers: Speakers::None,
        input: InputLimit::UpTo(std::time::Duration::from_secs(5_218)),
    }
}

/// A Qwen3-ASR model file of the catalog stored as `quantization`.
pub fn qwen3_info(id: &str, size_class: Qwen3AsrSize, quantization: &str, size: u64) -> ModelInfo {
    info_of(
        id,
        size,
        false,
        ModelDescriptor {
            engine: EngineId::TranscribeCpp,
            family: ModelFamily::Qwen3Asr(size_class),
            format: ModelFormat::Gguf,
            weights: weights(Some(quantization)),
            profile: qwen3_profile(),
        },
    )
}

/// The weights named as the tests spell them: a quantization, or `encoder/decoder` dtypes.
pub fn weights(spelling: Option<&str>) -> Weights {
    match spelling {
        None => Weights::Quantized("f16".parse().unwrap()),
        Some(text) => match text.split_once('/') {
            Some((encoder, decoder)) => Weights::Onnx {
                encoder: encoder.parse::<OnnxDtype>().unwrap(),
                decoder: decoder.parse::<OnnxDtype>().unwrap(),
            },
            None => Weights::Quantized(text.parse().unwrap()),
        },
    }
}

pub fn family(name: &str) -> ModelFamily {
    ModelFamily::Whisper(name.parse::<WhisperSize>().unwrap())
}

pub fn descriptor(family_name: &str, weights: Weights, english: bool) -> ModelDescriptor {
    match weights {
        Weights::Quantized(_) => ModelDescriptor {
            engine: EngineId::WhisperCpp,
            family: family(family_name),
            format: ModelFormat::GgmlBin,
            weights,
            profile: whisper_profile(english, false),
        },
        Weights::Onnx { .. } => ModelDescriptor {
            engine: EngineId::Browser,
            family: family(family_name),
            format: ModelFormat::Onnx,
            weights,
            profile: ModelProfile::browser(if english {
                LanguageSupport::english_only()
            } else {
                LanguageSupport::Any
            }),
        },
    }
}

pub fn info_of(id: &str, size: u64, installed: bool, descriptor: ModelDescriptor) -> ModelInfo {
    ModelInfo::new(
        id.to_owned(),
        id.to_owned(),
        String::new(),
        size,
        installed,
        descriptor,
    )
}

/// A model of the catalog; `weights` is a quantization, or `encoder/decoder` dtypes for a browser model.
pub fn info(
    id: &str,
    family_name: &str,
    size: u64,
    weights_spelling: Option<&str>,
    english: bool,
) -> ModelInfo {
    info_of(
        id,
        size,
        false,
        descriptor(
            family_name.strip_suffix(".en").unwrap_or(family_name),
            weights(weights_spelling),
            english,
        ),
    )
}
