use super::license::{Attribution, ModelLicense};
use serde_with::{DeserializeFromStr, SerializeDisplay};
use std::{fmt, str::FromStr};
use strum::{Display, EnumString, VariantArray};

/// Whisper's model size class. Declaration order is ascending size and drives group ordering.
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
    VariantArray,
    SerializeDisplay,
    DeserializeFromStr,
)]
pub enum WhisperSize {
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

impl WhisperSize {
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

    /// The name people know it by.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Tiny => "Tiny",
            Self::Base => "Base",
            Self::Small => "Small",
            Self::Medium => "Medium",
            Self::LargeV1 => "Large v1",
            Self::LargeV2 => "Large v2",
            Self::LargeV3 => "Large v3",
            Self::LargeV3Turbo => "Large v3 Turbo",
        }
    }
}

/// NVIDIA Parakeet's checkpoints. Declaration order is the order they are listed in.
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
    VariantArray,
    SerializeDisplay,
    DeserializeFromStr,
)]
pub enum ParakeetVariant {
    #[strum(serialize = "parakeet-tdt-0.6b-v3")]
    TdtV3,
}

impl ParakeetVariant {
    #[must_use]
    pub const fn summary(self) -> &'static str {
        match self {
            Self::TdtV3 => "fast and accurate, 25 European languages",
        }
    }

    /// The name people know it by.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::TdtV3 => "Parakeet TDT v3",
        }
    }
}

/// A model family: the line of models a model belongs to, and its size class within it.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerializeDisplay, DeserializeFromStr,
)]
pub enum ModelFamily {
    Whisper(WhisperSize),
    Parakeet(ParakeetVariant),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("`{name}` is not a model family")]
pub struct UnknownFamily {
    pub name: String,
}

impl ModelFamily {
    /// Every family, in listing order.
    pub fn all() -> impl Iterator<Item = Self> {
        WhisperSize::VARIANTS
            .iter()
            .copied()
            .map(Self::Whisper)
            .chain(
                ParakeetVariant::VARIANTS
                    .iter()
                    .copied()
                    .map(Self::Parakeet),
            )
    }

    #[must_use]
    pub const fn summary(self) -> &'static str {
        match self {
            Self::Whisper(size) => size.summary(),
            Self::Parakeet(variant) => variant.summary(),
        }
    }

    /// The name people know it by.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Whisper(size) => size.label(),
            Self::Parakeet(variant) => variant.label(),
        }
    }
}

impl ModelFamily {
    /// Who made the family's models and under which terms they may be used.
    #[must_use]
    pub const fn attribution(self) -> Attribution {
        match self {
            Self::Whisper(_) => Attribution {
                license: ModelLicense::Mit,
                author: "OpenAI",
                work: "Whisper",
                origin_url: "https://github.com/openai/whisper",
                changes: None,
            },
            Self::Parakeet(ParakeetVariant::TdtV3) => Attribution {
                license: ModelLicense::CcBy4,
                author: "NVIDIA",
                work: "parakeet-tdt-0.6b-v3",
                origin_url: "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
                changes: Some(
                    "Converted to GGUF by handy-computer for transcribe.cpp; the quantized \
                     files also store the weights at reduced numeric precision. The weights \
                     were not retrained or fine-tuned.",
                ),
            },
        }
    }
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Whisper(size) => fmt::Display::fmt(size, f),
            Self::Parakeet(variant) => fmt::Display::fmt(variant, f),
        }
    }
}

impl FromStr for ModelFamily {
    type Err = UnknownFamily;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        WhisperSize::from_str(name)
            .map(Self::Whisper)
            .or_else(|_| ParakeetVariant::from_str(name).map(Self::Parakeet))
            .map_err(|_| UnknownFamily {
                name: name.to_owned(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families_are_stored_under_their_size_name() {
        let family = ModelFamily::Whisper(WhisperSize::LargeV3Turbo);
        assert_eq!(
            serde_json::to_string(&family).unwrap(),
            "\"large-v3-turbo\""
        );
        assert_eq!(
            serde_json::from_str::<ModelFamily>("\"tiny\"").unwrap(),
            ModelFamily::Whisper(WhisperSize::Tiny)
        );
    }

    #[test]
    fn a_parakeet_is_stored_under_its_checkpoint_name() {
        let family = ModelFamily::Parakeet(ParakeetVariant::TdtV3);
        assert_eq!(
            serde_json::to_string(&family).unwrap(),
            "\"parakeet-tdt-0.6b-v3\""
        );
        assert_eq!(
            serde_json::from_str::<ModelFamily>("\"parakeet-tdt-0.6b-v3\"").unwrap(),
            family
        );
    }

    #[test]
    fn parakeet_is_listed_after_every_whisper() {
        assert!(
            ModelFamily::Whisper(WhisperSize::LargeV3Turbo)
                < ModelFamily::Parakeet(ParakeetVariant::TdtV3)
        );
    }

    #[test]
    fn models_carry_the_terms_of_their_makers() {
        let whisper = ModelFamily::Whisper(WhisperSize::Tiny).attribution();
        assert_eq!(whisper.license, ModelLicense::Mit);
        assert_eq!(whisper.changes, None);
        let parakeet = ModelFamily::Parakeet(ParakeetVariant::TdtV3).attribution();
        assert_eq!(parakeet.license, ModelLicense::CcBy4);
        assert_eq!(parakeet.author, "NVIDIA");
        assert!(parakeet.changes.is_some());
    }

    #[test]
    fn every_family_is_listed_in_ascending_order() {
        let all: Vec<ModelFamily> = ModelFamily::all().collect();
        assert_eq!(all.len(), 9);
        assert!(all.is_sorted());
    }

    #[test]
    fn a_name_that_is_no_family_is_refused() {
        assert_eq!(
            "distil-large".parse::<ModelFamily>(),
            Err(UnknownFamily {
                name: "distil-large".to_owned()
            })
        );
        assert!(serde_json::from_str::<ModelFamily>("\"tiny.en\"").is_err());
    }

    #[test]
    fn larger_sizes_sort_after_smaller_ones() {
        assert!(ModelFamily::Whisper(WhisperSize::Tiny) < ModelFamily::Whisper(WhisperSize::Base));
        assert!(
            ModelFamily::Whisper(WhisperSize::LargeV3)
                < ModelFamily::Whisper(WhisperSize::LargeV3Turbo)
        );
    }
}
