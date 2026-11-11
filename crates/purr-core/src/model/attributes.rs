//! Typed attributes describing a Whisper model.

use serde_with::{DeserializeFromStr, SerializeDisplay};
use strum::{Display, EnumString};

/// Model size class. Declaration order is ascending size and drives group ordering.
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
pub enum ModelFamily {
    #[strum(serialize = "tiny")]
    Tiny,
    #[strum(serialize = "base")]
    Base,
    #[strum(serialize = "small")]
    Small,
    #[strum(serialize = "medium")]
    Medium,
    #[strum(serialize = "large-v1")]
    LargeV1,
    #[strum(serialize = "large-v2")]
    LargeV2,
    #[strum(serialize = "large-v3")]
    LargeV3,
    #[strum(serialize = "large-v3-turbo")]
    LargeV3Turbo,
}

impl ModelFamily {
    #[must_use]
    pub const fn summary(self) -> &'static str {
        match self {
            Self::Tiny => "fastest, lowest accuracy",
            Self::Base => "good balance of speed and accuracy",
            Self::Small => "good accuracy",
            Self::Medium => "high accuracy",
            Self::LargeV1 => "highest accuracy, original large model",
            Self::LargeV2 => "highest accuracy, improved large model",
            Self::LargeV3 => "most accurate",
            Self::LargeV3Turbo => "faster large model",
        }
    }
}

/// Languages a model can transcribe.
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
pub enum ModelLanguage {
    #[strum(serialize = "multilingual")]
    Multilingual,
    #[strum(serialize = "english-only")]
    EnglishOnly,
}

/// Weight precision of the model file.
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
pub enum Quantization {
    #[strum(serialize = "fp16")]
    Fp16,
    #[strum(serialize = "q5_0")]
    Q5_0,
    #[strum(serialize = "q5_1")]
    Q5_1,
    #[strum(serialize = "q8_0")]
    Q8_0,
}

/// Optional capabilities of a model beyond plain transcription.
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
pub enum ModelFeature {
    #[strum(serialize = "tdrz")]
    Tdrz,
}

impl ModelFeature {
    #[must_use]
    pub const fn summary(self) -> &'static str {
        match self {
            Self::Tdrz => "speaker turn detection (TinyDiarize)",
        }
    }
}
