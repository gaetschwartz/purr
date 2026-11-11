//! Typed attributes describing a Whisper model.

use serde_with::{DeserializeFromStr, SerializeDisplay};
use strum::{Display, EnumMessage, EnumString};

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
    EnumMessage,
    SerializeDisplay,
    DeserializeFromStr,
)]
pub enum ModelFamily {
    #[strum(serialize = "tiny", message = "fastest, lowest accuracy")]
    Tiny,
    #[strum(serialize = "base", message = "good balance of speed and accuracy")]
    Base,
    #[strum(serialize = "small", message = "good accuracy")]
    Small,
    #[strum(serialize = "medium", message = "high accuracy")]
    Medium,
    #[strum(serialize = "large-v1", message = "highest accuracy, original large")]
    LargeV1,
    #[strum(serialize = "large-v2", message = "highest accuracy, improved large")]
    LargeV2,
    #[strum(serialize = "large-v3", message = "most accurate")]
    LargeV3,
    #[strum(serialize = "large-v3-turbo", message = "faster large model")]
    LargeV3Turbo,
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
    EnumMessage,
    SerializeDisplay,
    DeserializeFromStr,
)]
pub enum ModelFeature {
    #[strum(serialize = "tdrz", message = "speaker turn detection (TinyDiarize)")]
    Tdrz,
}
