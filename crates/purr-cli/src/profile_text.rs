use purr_common::{
    language::support::LanguageSupport,
    model::profile::{ModelProfile, Speakers},
};

/// The languages of a model as the `models info` listing names them.
pub fn languages(support: &LanguageSupport) -> String {
    match support {
        LanguageSupport::Any => "multilingual".to_owned(),
        LanguageSupport::Only(tags) => {
            if *support == LanguageSupport::english_only() {
                "english-only".to_owned()
            } else {
                tags.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        }
    }
}

/// The features of a model as the `models info` listing names them.
pub fn features(profile: &ModelProfile) -> Vec<&'static str> {
    match profile.speakers {
        Speakers::None => Vec::new(),
        Speakers::TurnMarkers => vec!["tdrz"],
        Speakers::Labelled => vec!["speaker labels"],
    }
}

/// What a model offers beyond transcription, in a sentence part for the model list.
pub fn notes(profile: &ModelProfile) -> Vec<&'static str> {
    match profile.speakers {
        Speakers::None => Vec::new(),
        Speakers::TurnMarkers => vec!["speaker turn detection (TinyDiarize)"],
        Speakers::Labelled => vec!["speaker labels"],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::model::profile::{DetectionReport, HintEffect, InputLimit, Timing};

    fn profile(speakers: Speakers) -> ModelProfile {
        ModelProfile {
            languages: LanguageSupport::Any,
            language_hint: HintEffect::Steers,
            detection: DetectionReport::Reported,
            timing: Timing::Word,
            word_scores: true,
            speakers,
            input: InputLimit::Unbounded,
        }
    }

    #[test]
    fn the_languages_keep_the_names_the_listing_always_used() {
        assert_eq!(languages(&LanguageSupport::Any), "multilingual");
        assert_eq!(languages(&LanguageSupport::english_only()), "english-only");
    }

    #[test]
    fn other_language_lists_name_each_language() {
        let support = LanguageSupport::Only(
            ["ja", "pt-BR"]
                .iter()
                .map(|text| text.parse().unwrap())
                .collect(),
        );
        assert_eq!(languages(&support), "ja, pt-BR");
    }

    #[test]
    fn turn_markers_are_the_tdrz_feature() {
        assert_eq!(features(&profile(Speakers::None)), Vec::<&str>::new());
        assert_eq!(features(&profile(Speakers::TurnMarkers)), ["tdrz"]);
        assert_eq!(
            notes(&profile(Speakers::TurnMarkers)),
            ["speaker turn detection (TinyDiarize)"]
        );
    }
}
