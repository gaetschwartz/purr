//! The list the language picker offers.

use super::run::RunOutcome;
use crate::utils::language_label::language_label;
use purr_common::{
    language::{
        choice::LanguageChoice,
        known::{self, KnownLanguage},
        support::LanguageSupport,
        tag::LanguageTag,
    },
    model::profile::{DetectionReport, HintEffect, ModelProfile},
};

/// The picker's list for `query`: "Automatic" first, then every language whose name or code
/// starts a word matching the query, in Whisper's order.
#[must_use]
pub fn options(query: &str) -> Vec<LanguageChoice> {
    let query = query.trim().to_lowercase();
    let automatic = query.is_empty() || matches_automatic(&query);
    let languages = known::all()
        .iter()
        .filter(|language| query.is_empty() || matches_language(language, &query))
        .map(|language| LanguageChoice::Spoken(language.tag.clone()));
    automatic
        .then_some(LanguageChoice::Automatic)
        .into_iter()
        .chain(languages)
        .collect()
}

/// Whether the model finds the language itself, whatever it is told.
#[must_use]
pub fn ignores_hint(profile: &ModelProfile) -> bool {
    match profile.language_hint {
        HintEffect::Ignored => true,
        HintEffect::Steers | HintEffect::Required => false,
    }
}

/// Whether the model knows one language, so there is nothing to choose.
#[must_use]
pub fn is_locked(profile: &ModelProfile) -> bool {
    match profile.language_hint {
        HintEffect::Ignored => false,
        HintEffect::Steers | HintEffect::Required => profile.languages.single().is_some(),
    }
}

/// The picker's list for `query` with the model of `profile`; any language when no model is known.
#[must_use]
pub fn choices(profile: Option<&ModelProfile>, query: &str) -> Vec<LanguageChoice> {
    let Some(profile) = profile else {
        return options(query);
    };
    let query = query.trim().to_lowercase();
    let listed = |tags: &[LanguageTag]| -> Vec<LanguageChoice> {
        tags.iter()
            .cloned()
            .map(LanguageChoice::Spoken)
            .filter(|choice| query.is_empty() || matches_choice(choice, &query))
            .collect()
    };
    let automatic =
        (query.is_empty() || matches_automatic(&query)).then_some(LanguageChoice::Automatic);
    match (&profile.language_hint, &profile.languages) {
        (HintEffect::Ignored, _) => vec![LanguageChoice::Automatic],
        (HintEffect::Steers | HintEffect::Required, LanguageSupport::Any) => {
            let mut list = options(&query);
            if !profile.can_detect() {
                list.retain(|choice| *choice != LanguageChoice::Automatic);
            }
            list
        }
        (HintEffect::Steers | HintEffect::Required, LanguageSupport::Only(tags)) => {
            match profile.languages.single() {
                Some(tag) => vec![LanguageChoice::Spoken(tag.clone())],
                None => automatic
                    .filter(|_| profile.can_detect())
                    .into_iter()
                    .chain(listed(tags))
                    .collect(),
            }
        }
    }
}

/// How many spoken languages the model of `profile` offers; every known language without one.
#[must_use]
pub fn spoken_count(profile: Option<&ModelProfile>) -> usize {
    choices(profile, "")
        .iter()
        .filter(|choice| matches!(choice, LanguageChoice::Spoken(_)))
        .count()
}

/// What "Automatic" does with `profile`: it detects the language, and says which when the model
/// reports it.
#[must_use]
pub fn automatic_hint(profile: Option<&ModelProfile>) -> &'static str {
    match profile.map(|profile| profile.detection) {
        Some(DetectionReport::Silent) => "Detected, not reported",
        Some(DetectionReport::Reported) | None => "Detected from the start",
    }
}

/// What the language pill adds to its value: a detail beside its caption, and a tooltip.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PillNotes {
    pub detail: Option<String>,
    pub hint: Option<String>,
}

/// The notes of the pill showing `choice` for the model of `profile`, called `name`.
#[must_use]
pub fn pill_notes(
    choice: &LanguageChoice,
    profile: Option<&ModelProfile>,
    name: &str,
) -> PillNotes {
    if profile.is_some_and(ignores_hint) {
        return PillNotes {
            detail: None,
            hint: Some(format!("{name} detects the language itself")),
        };
    }
    match choice {
        LanguageChoice::Spoken(_) if profile.is_some_and(is_locked) => PillNotes {
            detail: Some(format!("{} only", choice.label())),
            hint: None,
        },
        LanguageChoice::Spoken(_) => PillNotes::default(),
        LanguageChoice::Automatic => PillNotes {
            detail: None,
            hint: Some(format!("Automatic · {}", automatic_hint(profile))),
        },
    }
}

/// Whether a run asked for `choice` with the model of `profile` will say which language it found.
#[must_use]
pub fn reports_language(choice: &LanguageChoice, profile: &ModelProfile) -> bool {
    match (choice, profile.detection) {
        (LanguageChoice::Automatic, DetectionReport::Reported) => true,
        (LanguageChoice::Automatic, DetectionReport::Silent)
        | (LanguageChoice::Spoken(_), DetectionReport::Reported | DetectionReport::Silent) => false,
    }
}

/// The language line of a run: the language that applied, or what is known of it while the model
/// finds it by itself.
#[must_use]
pub fn run_language_text(
    choice: &LanguageChoice,
    language: Option<LanguageTag>,
    outcome: &RunOutcome,
    detection: DetectionReport,
) -> String {
    match (choice, language, outcome, detection) {
        (LanguageChoice::Automatic, Some(code), _, _) => {
            format!("{} (detected)", language_label(code.as_str()))
        }
        (LanguageChoice::Automatic, None, RunOutcome::Running, DetectionReport::Reported) => {
            "Language not known yet".to_owned()
        }
        (
            LanguageChoice::Automatic,
            None,
            RunOutcome::Running | RunOutcome::Completed(_) | RunOutcome::Failed(_),
            DetectionReport::Silent,
        )
        | (
            LanguageChoice::Automatic,
            None,
            RunOutcome::Completed(_) | RunOutcome::Failed(_),
            DetectionReport::Reported,
        ) => "Language not reported".to_owned(),
        (LanguageChoice::Spoken(_), _, _, _) => choice.label(),
    }
}

/// The choice as the model of `profile` takes it.
#[must_use]
pub fn for_model(choice: LanguageChoice, profile: Option<&ModelProfile>) -> LanguageChoice {
    profile.map_or_else(
        || choice.clone(),
        |profile| choice.clone().for_profile(profile),
    )
}

fn matches_language(language: &KnownLanguage, query: &str) -> bool {
    matches_name(language.tag.as_str(), language.name, query)
}

fn matches_choice(choice: &LanguageChoice, query: &str) -> bool {
    match choice {
        LanguageChoice::Automatic => matches_automatic(query),
        LanguageChoice::Spoken(tag) => matches_name(
            &tag.as_str().to_lowercase(),
            &choice.label().to_lowercase(),
            query,
        ),
    }
}

fn matches_automatic(query: &str) -> bool {
    ["automatic", "detect"]
        .iter()
        .any(|word| word.starts_with(query))
}

fn matches_name(tag: &str, name: &str, query: &str) -> bool {
    tag == query
        || name.split_whitespace().any(|word| word.starts_with(query))
        || name.starts_with(query)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spoken(text: &str) -> LanguageChoice {
        LanguageChoice::Spoken(text.parse().unwrap())
    }

    #[test]
    fn an_empty_query_lists_automatic_then_every_language() {
        let all = options("");
        assert_eq!(all.first(), Some(&LanguageChoice::Automatic));
        assert_eq!(all.get(1), Some(&spoken("en")));
        assert_eq!(all.len(), known::all().len() + 1);
    }

    fn profile(hint: HintEffect, languages: LanguageSupport) -> ModelProfile {
        let mut profile = ModelProfile::browser(languages);
        profile.language_hint = hint;
        profile
    }

    #[test]
    fn a_model_that_knows_one_language_offers_only_it() {
        let english = profile(HintEffect::Steers, LanguageSupport::english_only());
        assert!(is_locked(&english));
        assert!(!ignores_hint(&english));
        assert_eq!(choices(Some(&english), "fr"), vec![spoken("en")]);
    }

    fn parakeet() -> ModelProfile {
        crate::model_catalog::test_support::parakeet_profile()
    }

    #[test]
    fn a_model_that_finds_the_language_itself_ignores_the_hint() {
        assert!(ignores_hint(&parakeet()));
        assert!(!is_locked(&parakeet()));
        let ignored = profile(HintEffect::Ignored, LanguageSupport::Any);
        assert!(ignores_hint(&ignored));
    }

    #[test]
    fn a_model_the_hint_steers_does_not_ignore_it() {
        let whisper = crate::model_catalog::test_support::whisper_profile(false, false);
        assert!(!ignores_hint(&whisper));
    }

    #[test]
    fn only_automatic_with_a_model_that_reports_it_waits_for_a_detected_language() {
        let whisper = crate::model_catalog::test_support::whisper_profile(false, false);
        assert!(reports_language(&LanguageChoice::Automatic, &whisper));
        assert!(!reports_language(&spoken("fr"), &whisper));
        assert!(!reports_language(&LanguageChoice::Automatic, &parakeet()));
    }

    #[test]
    fn a_run_with_a_model_that_never_names_the_language_says_so_from_the_start() {
        let say = |detection, outcome: &RunOutcome| {
            run_language_text(&LanguageChoice::Automatic, None, outcome, detection)
        };
        assert_eq!(
            say(DetectionReport::Silent, &RunOutcome::Running),
            "Language not reported"
        );
        assert_eq!(
            say(DetectionReport::Reported, &RunOutcome::Running),
            "Language not known yet"
        );
        assert_eq!(
            say(
                DetectionReport::Reported,
                &RunOutcome::Failed(crate::transcribe::run::RunFailure {
                    message: String::new(),
                    hint: None
                })
            ),
            "Language not reported"
        );
    }

    #[test]
    fn a_detected_language_is_named_and_a_chosen_one_is_not_called_detected() {
        let french = Some("fr".parse().unwrap());
        assert_eq!(
            run_language_text(
                &LanguageChoice::Automatic,
                french.clone(),
                &RunOutcome::Running,
                DetectionReport::Reported
            ),
            "French (detected)"
        );
        assert_eq!(
            run_language_text(
                &spoken("fr"),
                french,
                &RunOutcome::Running,
                DetectionReport::Reported
            ),
            "French"
        );
    }

    #[test]
    fn automatic_says_it_detects_and_whether_the_language_is_reported() {
        let whisper = crate::model_catalog::test_support::whisper_profile(false, false);
        assert_eq!(automatic_hint(Some(&whisper)), "Detected from the start");
        assert_eq!(automatic_hint(None), "Detected from the start");
        assert_eq!(automatic_hint(Some(&parakeet())), "Detected, not reported");
    }

    #[test]
    fn the_pill_of_a_model_that_ignores_the_hint_says_it_detects_the_language_itself() {
        let notes = pill_notes(
            &LanguageChoice::Automatic,
            Some(&parakeet()),
            "Parakeet TDT v3",
        );
        assert_eq!(
            notes,
            PillNotes {
                detail: None,
                hint: Some("Parakeet TDT v3 detects the language itself".to_owned())
            }
        );
        let saved = pill_notes(&spoken("fr"), Some(&parakeet()), "Parakeet TDT v3");
        assert_eq!(saved, notes);
    }

    #[test]
    fn the_pill_of_a_model_that_knows_one_language_names_it() {
        let english = profile(HintEffect::Steers, LanguageSupport::english_only());
        assert_eq!(
            pill_notes(&spoken("en"), Some(&english), "Tiny"),
            PillNotes {
                detail: Some("English only".to_owned()),
                hint: None
            }
        );
    }

    #[test]
    fn the_pill_of_a_model_the_hint_steers_explains_automatic_only() {
        let whisper = crate::model_catalog::test_support::whisper_profile(false, false);
        assert_eq!(
            pill_notes(&LanguageChoice::Automatic, Some(&whisper), "Base").hint,
            Some("Automatic · Detected from the start".to_owned())
        );
        assert_eq!(
            pill_notes(&spoken("fr"), Some(&whisper), "Base"),
            PillNotes::default()
        );
        assert_eq!(
            pill_notes(&spoken("fr"), None, "This model"),
            PillNotes::default()
        );
    }

    #[test]
    fn a_model_that_needs_the_language_does_not_offer_detection() {
        let required = profile(HintEffect::Required, LanguageSupport::Any);
        assert!(!is_locked(&required));
        let offered = choices(Some(&required), "");
        assert_eq!(offered.first(), Some(&spoken("en")));
        assert_eq!(offered.len(), known::all().len());
    }

    #[test]
    fn a_model_with_a_few_languages_offers_those_and_detection() {
        let few = profile(
            HintEffect::Steers,
            LanguageSupport::Only(
                ["en", "fr"]
                    .iter()
                    .map(|tag| tag.parse().unwrap())
                    .collect(),
            ),
        );
        assert!(!is_locked(&few));
        assert_eq!(
            choices(Some(&few), ""),
            vec![LanguageChoice::Automatic, spoken("en"), spoken("fr")]
        );
    }

    #[test]
    fn searching_the_languages_of_a_model_with_a_few_filters_them() {
        let few = profile(
            HintEffect::Steers,
            LanguageSupport::Only(
                ["en", "fr"]
                    .iter()
                    .map(|tag| tag.parse().unwrap())
                    .collect(),
            ),
        );
        assert_eq!(choices(Some(&few), "fr"), vec![spoken("fr")]);
        assert_eq!(spoken_count(Some(&few)), 2);
    }

    #[test]
    fn an_unknown_model_offers_every_language() {
        assert_eq!(choices(None, "fre"), options("fre"));
    }

    #[test]
    fn search_matches_word_starts_and_codes() {
        assert_eq!(options("fre"), vec![spoken("fr")]);
        assert_eq!(options("CREOLE"), vec![spoken("ht")]);
        assert_eq!(options("de"), vec![LanguageChoice::Automatic, spoken("de")]);
        assert_eq!(options("hebr"), vec![spoken("he")]);
        assert_eq!(options("zzz"), Vec::<LanguageChoice>::new());
    }
}
