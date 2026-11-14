use crate::language::support::LanguageSupport;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// What choosing a spoken language does to a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HintEffect {
    /// The model decodes in the chosen language
    Steers,
    /// The model finds the language itself and takes no hint
    Ignored,
    /// The model cannot find the language itself and needs the choice
    Required,
}

/// Whether a model says which language it recognised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DetectionReport {
    Reported,
    Silent,
}

/// The finest timing a model produces, coarsest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Timing {
    None,
    Segment,
    Word,
    /// Parts of words; the words are found by joining them
    Token,
}

impl Timing {
    /// Whether the words of a transcript can carry times.
    #[must_use]
    pub const fn times_words(self) -> bool {
        match self {
            Self::None | Self::Segment => false,
            Self::Word | Self::Token => true,
        }
    }
}

/// What a model says about who speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Speakers {
    None,
    /// The text marks where the speaker changes
    TurnMarkers,
    /// The segments name their speaker
    Labelled,
}

/// The longest audio a model takes in one piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InputLimit {
    Unbounded,
    UpTo(#[serde(with = "seconds")] Duration),
}

mod seconds {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(duration.as_secs())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
        u64::deserialize(deserializer).map(Duration::from_secs)
    }
}

/// What a model can do beyond plain transcription, as its engine declares it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProfile {
    pub languages: LanguageSupport,
    pub language_hint: HintEffect,
    pub detection: DetectionReport,
    pub timing: Timing,
    pub word_scores: bool,
    pub speakers: Speakers,
    pub input: InputLimit,
}

impl ModelProfile {
    /// What the browser's Whisper reports: segments with times, and nothing about the words.
    #[must_use]
    pub fn browser(languages: LanguageSupport) -> Self {
        Self {
            languages,
            language_hint: HintEffect::Steers,
            detection: DetectionReport::Reported,
            timing: Timing::Segment,
            word_scores: false,
            speakers: Speakers::None,
            input: InputLimit::Unbounded,
        }
    }

    /// Whether the model reports a score or a time for the words.
    #[must_use]
    pub const fn reports_word_detail(&self) -> bool {
        self.word_scores || self.timing.times_words()
    }

    /// Whether "automatic" is a language the model can be given.
    #[must_use]
    pub const fn can_detect(&self) -> bool {
        match self.language_hint {
            HintEffect::Steers | HintEffect::Ignored => true,
            HintEffect::Required => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn timing_is_ordered_from_nothing_to_the_finest() {
        assert!(Timing::None < Timing::Segment);
        assert!(Timing::Segment < Timing::Word);
        assert!(Timing::Word < Timing::Token);
        assert!(!Timing::Segment.times_words());
        assert!(Timing::Token.times_words());
    }

    #[test]
    fn a_profile_is_read_from_its_manifest_form() {
        let profile: ModelProfile = serde_json::from_str(
            r#"{"languages":{"only":["en-US"]},"language_hint":"steers","detection":"silent",
            "timing":"token","word_scores":false,"speakers":"labelled","input":{"up-to":30}}"#,
        )
        .unwrap();
        assert_eq!(
            profile,
            ModelProfile {
                languages: LanguageSupport::Only(Arc::from(["en-US".parse().unwrap()])),
                language_hint: HintEffect::Steers,
                detection: DetectionReport::Silent,
                timing: Timing::Token,
                word_scores: false,
                speakers: Speakers::Labelled,
                input: InputLimit::UpTo(Duration::from_secs(30)),
            }
        );
    }

    #[test]
    fn a_field_the_profile_does_not_know_is_refused() {
        let unknown = r#"{"languages":"any","language_hint":"steers","detection":"reported",
            "timing":"word","word_scores":true,"speakers":"none","input":"unbounded",
            "translate":true}"#;
        assert!(serde_json::from_str::<ModelProfile>(unknown).is_err());
    }

    #[test]
    fn the_browser_reports_no_word_detail() {
        let profile = ModelProfile::browser(LanguageSupport::Any);
        assert!(!profile.reports_word_detail());
        assert!(profile.can_detect());
        assert_eq!(profile.timing, Timing::Segment);
    }

    #[test]
    fn a_model_that_needs_the_language_cannot_detect_it() {
        let mut profile = ModelProfile::browser(LanguageSupport::Any);
        profile.language_hint = HintEffect::Required;
        assert!(!profile.can_detect());
    }
}
