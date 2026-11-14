//! The list the language picker offers.

use purr_common::{
    language::{
        choice::LanguageChoice,
        known::{self, KnownLanguage},
        support::LanguageSupport,
    },
    model::profile::{HintEffect, ModelProfile},
};

/// The picker's list for `query`: "Automatic" first, then every language whose name or code
/// starts a word matching the query, in Whisper's order.
#[must_use]
pub fn options(query: &str) -> Vec<LanguageChoice> {
    let query = query.trim().to_lowercase();
    let automatic = query.is_empty()
        || ["automatic", "detect"]
            .iter()
            .any(|word| word.starts_with(&query));
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

/// Whether the model leaves nothing to choose: it knows one language, or finds it itself.
#[must_use]
pub fn is_locked(profile: &ModelProfile) -> bool {
    match profile.language_hint {
        HintEffect::Ignored => true,
        HintEffect::Steers | HintEffect::Required => profile.languages.single().is_some(),
    }
}

/// The picker's list for `query` with the model of `profile`; any language when no model is known.
#[must_use]
pub fn choices(profile: Option<&ModelProfile>, query: &str) -> Vec<LanguageChoice> {
    let Some(profile) = profile else {
        return options(query);
    };
    match profile.language_hint {
        HintEffect::Ignored => vec![LanguageChoice::Automatic],
        HintEffect::Steers | HintEffect::Required => {
            let mut list = match &profile.languages {
                LanguageSupport::Any => options(query),
                LanguageSupport::Only(tags) => match profile.languages.single() {
                    Some(tag) => vec![LanguageChoice::Spoken(tag.clone())],
                    None => std::iter::once(LanguageChoice::Automatic)
                        .chain(tags.iter().cloned().map(LanguageChoice::Spoken))
                        .collect(),
                },
            };
            if !profile.can_detect() {
                list.retain(|choice| *choice != LanguageChoice::Automatic);
            }
            list
        }
    }
}

/// What the picker says about a model that restricts the choice.
#[must_use]
pub fn note(profile: &ModelProfile) -> Option<String> {
    match profile.language_hint {
        HintEffect::Ignored => Some("This model detects the language itself.".to_owned()),
        HintEffect::Steers | HintEffect::Required => profile.languages.single().map(|tag| {
            format!(
                "This model only transcribes {}.",
                LanguageChoice::Spoken(tag.clone()).label()
            )
        }),
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
    language.tag.as_str() == query
        || language
            .name
            .split_whitespace()
            .any(|word| word.starts_with(query))
        || language.name.starts_with(query)
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
        assert_eq!(choices(Some(&english), "fr"), vec![spoken("en")]);
        assert_eq!(
            note(&english),
            Some("This model only transcribes English.".to_owned())
        );
    }

    #[test]
    fn a_model_without_a_hint_offers_only_detection() {
        let ignored = profile(HintEffect::Ignored, LanguageSupport::Any);
        assert!(is_locked(&ignored));
        assert_eq!(choices(Some(&ignored), ""), vec![LanguageChoice::Automatic]);
        assert_eq!(
            note(&ignored),
            Some("This model detects the language itself.".to_owned())
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
        assert_eq!(note(&few), None);
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
