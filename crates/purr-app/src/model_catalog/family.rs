use purr_common::model::family::{ModelFamily, WhisperSize};

/// The line of models a family belongs to, which the lists name before its sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelLine {
    Whisper,
    Parakeet,
    Qwen3Asr,
}

impl ModelLine {
    #[must_use]
    pub const fn of(family: ModelFamily) -> Self {
        match family {
            ModelFamily::Whisper(_) => Self::Whisper,
            ModelFamily::Parakeet(_) => Self::Parakeet,
            ModelFamily::Qwen3Asr(_) => Self::Qwen3Asr,
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Whisper => "Whisper",
            Self::Parakeet => "Parakeet",
            Self::Qwen3Asr => "Qwen3-ASR",
        }
    }
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
    Parakeet,
    Qwen3Asr,
    /// Models whose family this app does not know.
    Other,
}

impl ModelGroup {
    /// The sections in the order the list shows them.
    pub const ORDER: [Self; 9] = [
        Self::Turbo,
        Self::Large,
        Self::Medium,
        Self::Small,
        Self::Base,
        Self::Tiny,
        Self::Parakeet,
        Self::Qwen3Asr,
        Self::Other,
    ];

    /// The list section a family belongs to.
    #[must_use]
    pub const fn of(family: ModelFamily) -> Self {
        match family {
            ModelFamily::Whisper(size) => match size {
                WhisperSize::Tiny => Self::Tiny,
                WhisperSize::Base => Self::Base,
                WhisperSize::Small => Self::Small,
                WhisperSize::Medium => Self::Medium,
                WhisperSize::LargeV1 | WhisperSize::LargeV2 | WhisperSize::LargeV3 => Self::Large,
                WhisperSize::LargeV3Turbo => Self::Turbo,
            },
            ModelFamily::Parakeet(_) => Self::Parakeet,
            ModelFamily::Qwen3Asr(_) => Self::Qwen3Asr,
        }
    }

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
            Self::Parakeet => "Parakeet",
            Self::Qwen3Asr => "Qwen3-ASR",
            Self::Other => "Other models",
        }
    }
}

/// Whether a newer family replaces `family`.
#[must_use]
pub const fn superseded(family: ModelFamily) -> bool {
    match family {
        ModelFamily::Whisper(size) => match size {
            WhisperSize::LargeV1 | WhisperSize::LargeV2 => true,
            WhisperSize::Tiny
            | WhisperSize::Base
            | WhisperSize::Small
            | WhisperSize::Medium
            | WhisperSize::LargeV3
            | WhisperSize::LargeV3Turbo => false,
        },
        ModelFamily::Parakeet(_) | ModelFamily::Qwen3Asr(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::model::family::Qwen3AsrSize;

    fn whisper(size: WhisperSize) -> ModelFamily {
        ModelFamily::Whisper(size)
    }

    #[test]
    fn large_releases_share_a_group_but_turbo_has_its_own() {
        assert_eq!(
            ModelGroup::of(whisper(WhisperSize::LargeV1)),
            ModelGroup::Large
        );
        assert_eq!(
            ModelGroup::of(whisper(WhisperSize::LargeV2)),
            ModelGroup::Large
        );
        assert_eq!(
            ModelGroup::of(whisper(WhisperSize::LargeV3)),
            ModelGroup::Large
        );
        assert_eq!(
            ModelGroup::of(whisper(WhisperSize::LargeV3Turbo)),
            ModelGroup::Turbo
        );
    }

    #[test]
    fn qwen3_asr_is_listed_after_parakeet_in_a_section_of_its_own() {
        let family = ModelFamily::Qwen3Asr(Qwen3AsrSize::B0_6);
        assert_eq!(ModelGroup::of(family), ModelGroup::Qwen3Asr);
        assert_eq!(ModelLine::of(family), ModelLine::Qwen3Asr);
        assert!(!superseded(family));
        let position = |group| ModelGroup::ORDER.iter().position(|listed| *listed == group);
        assert!(position(ModelGroup::Parakeet) < position(ModelGroup::Qwen3Asr));
        assert!(position(ModelGroup::Qwen3Asr) < position(ModelGroup::Other));
    }

    #[test]
    fn only_the_old_large_families_are_superseded() {
        assert!(superseded(whisper(WhisperSize::LargeV1)));
        assert!(superseded(whisper(WhisperSize::LargeV2)));
        assert!(!superseded(whisper(WhisperSize::LargeV3)));
        assert!(!superseded(whisper(WhisperSize::Tiny)));
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
