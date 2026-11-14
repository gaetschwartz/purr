use super::format::ModelFormat;
use serde_with::{DeserializeFromStr, SerializeDisplay};
use strum::{Display, EnumString};

/// The speech recognition software that runs a model.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Display,
    EnumString,
    SerializeDisplay,
    DeserializeFromStr,
)]
pub enum EngineId {
    #[strum(serialize = "whisper-cpp")]
    WhisperCpp,
    #[strum(serialize = "transcribe-cpp")]
    TranscribeCpp,
    #[strum(serialize = "browser")]
    Browser,
}

impl EngineId {
    /// The file formats the engine loads.
    #[must_use]
    pub const fn formats(self) -> &'static [ModelFormat] {
        match self {
            Self::WhisperCpp => &[ModelFormat::GgmlBin],
            Self::TranscribeCpp => &[ModelFormat::Gguf],
            Self::Browser => &[ModelFormat::Onnx],
        }
    }

    /// The engine as a sentence names it.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::WhisperCpp => "whisper.cpp engine",
            Self::TranscribeCpp => "transcribe.cpp engine",
            Self::Browser => "browser engine",
        }
    }

    #[must_use]
    pub fn supports(self, format: ModelFormat) -> bool {
        self.formats().contains(&format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_engine_loads_its_own_format_only() {
        assert!(EngineId::WhisperCpp.supports(ModelFormat::GgmlBin));
        assert!(!EngineId::WhisperCpp.supports(ModelFormat::Gguf));
        assert!(EngineId::TranscribeCpp.supports(ModelFormat::Gguf));
        assert!(!EngineId::TranscribeCpp.supports(ModelFormat::Onnx));
        assert!(EngineId::Browser.supports(ModelFormat::Onnx));
        assert!(!EngineId::Browser.supports(ModelFormat::GgmlBin));
    }

    #[test]
    fn engines_are_stored_by_name() {
        assert_eq!(
            serde_json::to_string(&EngineId::TranscribeCpp).unwrap(),
            "\"transcribe-cpp\""
        );
        assert_eq!(
            serde_json::from_str::<EngineId>("\"whisper-cpp\"").unwrap(),
            EngineId::WhisperCpp
        );
        assert!(serde_json::from_str::<EngineId>("\"vosk\"").is_err());
    }
}
