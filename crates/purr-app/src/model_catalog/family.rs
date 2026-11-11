use std::str::FromStr;
use strum::EnumString;

/// A Whisper model size class, as the platforms name it in the `family` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString)]
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

const ENGLISH_SUFFIX: &str = ".en";

impl ModelFamily {
    /// The family named by a platform attribute; web manifests append `.en` for English-only ones.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        Self::from_str(raw.strip_suffix(ENGLISH_SUFFIX).unwrap_or(raw)).ok()
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

    /// The list section the family belongs to.
    #[must_use]
    pub const fn group(self) -> ModelGroup {
        match self {
            Self::Tiny => ModelGroup::Tiny,
            Self::Base => ModelGroup::Base,
            Self::Small => ModelGroup::Small,
            Self::Medium => ModelGroup::Medium,
            Self::LargeV1 | Self::LargeV2 | Self::LargeV3 => ModelGroup::Large,
            Self::LargeV3Turbo => ModelGroup::Turbo,
        }
    }

    /// A caveat that only applies to this family, if any.
    #[must_use]
    pub const fn note(self) -> Option<&'static str> {
        match self {
            Self::LargeV1 => Some("First large model, superseded by v3."),
            Self::LargeV2 => Some("Previous large model. Use it to match older transcripts."),
            Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::LargeV3
            | Self::LargeV3Turbo => None,
        }
    }
}

/// Speed and accuracy on a scale of one to five dots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ratings {
    pub speed: u8,
    pub accuracy: u8,
}

/// A section of the model list: related families shown together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelGroup {
    Turbo,
    Large,
    Medium,
    Small,
    Base,
    Tiny,
    /// Models whose family this app does not know.
    Other,
}

impl ModelGroup {
    /// The sections in the order the list shows them.
    pub const ORDER: [Self; 7] = [
        Self::Turbo,
        Self::Large,
        Self::Medium,
        Self::Small,
        Self::Base,
        Self::Tiny,
        Self::Other,
    ];

    /// The section title.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Turbo => "Large v3 Turbo",
            Self::Large => "Large",
            Self::Medium => "Medium",
            Self::Small => "Small",
            Self::Base => "Base",
            Self::Tiny => "Tiny",
            Self::Other => "Other models",
        }
    }

    /// What the group trades off, in plain words.
    #[must_use]
    pub const fn hint(self) -> &'static str {
        match self {
            Self::Turbo => "Nearly Large v3 accuracy at several times the speed.",
            Self::Large => "Most accurate. Slow without a GPU.",
            Self::Medium => "Accurate but slower. Best with a GPU.",
            Self::Small => "Balanced. Copes better with accents and background noise.",
            Self::Base => "Very fast. Handles everyday recordings well.",
            Self::Tiny => "Fastest. Good for clear speech and quick drafts.",
            Self::Other => "Models this version of Purr has no description for.",
        }
    }

    /// The speed and accuracy dots; `None` when the size class is unknown.
    #[must_use]
    pub const fn ratings(self) -> Option<Ratings> {
        match self {
            Self::Tiny => Some(Ratings {
                speed: 5,
                accuracy: 1,
            }),
            Self::Base => Some(Ratings {
                speed: 4,
                accuracy: 2,
            }),
            Self::Small => Some(Ratings {
                speed: 3,
                accuracy: 3,
            }),
            Self::Medium => Some(Ratings {
                speed: 2,
                accuracy: 4,
            }),
            Self::Large => Some(Ratings {
                speed: 1,
                accuracy: 5,
            }),
            Self::Turbo => Some(Ratings {
                speed: 4,
                accuracy: 5,
            }),
            Self::Other => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_suffixed_web_families_parse_like_desktop_ones() {
        assert_eq!(ModelFamily::parse("tiny"), Some(ModelFamily::Tiny));
        assert_eq!(ModelFamily::parse("tiny.en"), Some(ModelFamily::Tiny));
        assert_eq!(
            ModelFamily::parse("large-v3-turbo"),
            Some(ModelFamily::LargeV3Turbo)
        );
    }

    #[test]
    fn unknown_families_do_not_parse() {
        assert_eq!(ModelFamily::parse("distil-large"), None);
        assert_eq!(ModelFamily::parse(""), None);
    }

    #[test]
    fn large_releases_share_a_group_but_turbo_has_its_own() {
        assert_eq!(ModelFamily::LargeV1.group(), ModelGroup::Large);
        assert_eq!(ModelFamily::LargeV2.group(), ModelGroup::Large);
        assert_eq!(ModelFamily::LargeV3.group(), ModelGroup::Large);
        assert_eq!(ModelFamily::LargeV3Turbo.group(), ModelGroup::Turbo);
    }

    #[test]
    fn every_known_group_has_ratings_and_other_does_not() {
        for group in ModelGroup::ORDER {
            assert_eq!(
                group.ratings().is_some(),
                group != ModelGroup::Other,
                "{group:?}"
            );
        }
    }

    #[test]
    fn the_order_lists_each_group_once() {
        let mut seen = std::collections::HashSet::new();
        assert!(
            ModelGroup::ORDER
                .into_iter()
                .all(|group| seen.insert(group))
        );
    }
}
