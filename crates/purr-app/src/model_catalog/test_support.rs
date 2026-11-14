//! Models for the tests of the catalog, built the way the platforms describe them.

use purr_common::{
    language::support::LanguageSupport,
    model::{
        descriptor::{ModelDescriptor, Weights},
        engine::EngineId,
        family::{ModelFamily, WhisperSize},
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
