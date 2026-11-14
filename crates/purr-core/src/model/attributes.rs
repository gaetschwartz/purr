//! Typed attributes describing a Whisper model.

use serde_with::{DeserializeFromStr, SerializeDisplay};
use strum::{Display, EnumString};

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
