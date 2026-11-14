use serde_with::{DeserializeFromStr, SerializeDisplay};
use std::{fmt, str::FromStr};
use strum::{Display, EnumString};

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

/// A model family: the line of models a model belongs to, and its size class within it.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerializeDisplay, DeserializeFromStr,
)]
pub enum ModelFamily {
    Whisper(WhisperSize),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("`{name}` is not a model family")]
pub struct UnknownFamily {
    pub name: String,
}

impl ModelFamily {
    #[must_use]
    pub const fn summary(self) -> &'static str {
        match self {
            Self::Whisper(size) => size.summary(),
        }
    }

    /// The name people know it by.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Whisper(size) => size.label(),
        }
    }
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Whisper(size) => fmt::Display::fmt(size, f),
        }
    }
}

impl FromStr for ModelFamily {
    type Err = UnknownFamily;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        WhisperSize::from_str(name)
            .map(Self::Whisper)
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
