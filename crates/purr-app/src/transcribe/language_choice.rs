//! The spoken language: detected, or one of Whisper's languages.

use crate::utils::language_label::capitalized;
use purr_common::Language;

/// What the user asked for as the spoken language.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LanguageChoice {
    Automatic,
    Spoken(Language),
}

impl LanguageChoice {
    /// Reads the `ui.language` setting; a code this version does not know means detection.
    #[must_use]
    pub fn from_setting(code: &str) -> Self {
        match code.parse::<Language>() {
            Ok(Language::Auto) | Err(()) => Self::Automatic,
            Ok(language) => Self::Spoken(language),
        }
    }

    /// The value to store in the `ui.language` setting.
    #[must_use]
    pub fn setting(&self) -> &'static str {
        match self {
            Self::Automatic => Language::Auto.code(),
            Self::Spoken(language) => language.code(),
        }
    }

    /// The name shown in the pill and the list.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Automatic => "Automatic".to_owned(),
            Self::Spoken(language) => capitalized(language.name()),
        }
    }

    /// What applies with a model that may only know English: English-only models force English.
    #[must_use]
    pub fn for_model(self, english_only: bool) -> Self {
        if english_only {
            Self::Spoken(Language::En)
        } else {
            self
        }
    }

    /// The language to send with a transcription request; `None` asks the platform to detect it.
    #[must_use]
    pub fn request_code(&self) -> Option<String> {
        match self {
            Self::Automatic => None,
            Self::Spoken(language) => Some(language.code().to_owned()),
        }
    }
}

/// The picker's list for `query`: "Automatic" first, then every language whose name or code
/// starts a word matching the query, in Whisper's order.
#[must_use]
pub fn options(query: &str) -> Vec<LanguageChoice> {
    let query = query.trim().to_lowercase();
    let automatic = query.is_empty()
        || ["automatic", "detect"]
            .iter()
            .any(|word| word.starts_with(&query));
    let languages = Language::VARIANTS
        .iter()
        .filter(|language| **language != Language::Auto)
        .filter(|language| query.is_empty() || matches_language(language, &query))
        .cloned()
        .map(LanguageChoice::Spoken);
    automatic
        .then_some(LanguageChoice::Automatic)
        .into_iter()
        .chain(languages)
        .collect()
}

fn matches_language(language: &Language, query: &str) -> bool {
    language.code() == query
        || language
            .name()
            .split_whitespace()
            .any(|word| word.starts_with(query))
        || language.name().starts_with(query)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_setting_round_trips() {
        assert_eq!(
            LanguageChoice::from_setting("auto"),
            LanguageChoice::Automatic
        );
        assert_eq!(
            LanguageChoice::from_setting("fr"),
            LanguageChoice::Spoken(Language::Fr)
        );
        assert_eq!(LanguageChoice::Spoken(Language::Fr).setting(), "fr");
        assert_eq!(LanguageChoice::Automatic.setting(), "auto");
        assert_eq!(
            LanguageChoice::from_setting("xx"),
            LanguageChoice::Automatic
        );
    }

    #[test]
    fn labels_are_capitalized_names() {
        assert_eq!(LanguageChoice::Spoken(Language::De).label(), "German");
        assert_eq!(LanguageChoice::Automatic.label(), "Automatic");
    }

    #[test]
    fn english_only_models_force_english_in_the_request() {
        assert_eq!(
            LanguageChoice::Automatic.for_model(true).request_code(),
            Some("en".to_owned())
        );
        assert_eq!(
            LanguageChoice::Spoken(Language::Fr)
                .for_model(true)
                .request_code(),
            Some("en".to_owned())
        );
        assert_eq!(
            LanguageChoice::Automatic.for_model(false).request_code(),
            None
        );
        assert_eq!(
            LanguageChoice::Spoken(Language::Fr)
                .for_model(false)
                .request_code(),
            Some("fr".to_owned())
        );
    }

    #[test]
    fn an_empty_query_lists_automatic_then_every_language() {
        let all = options("");
        assert_eq!(all.first(), Some(&LanguageChoice::Automatic));
        assert_eq!(all.get(1), Some(&LanguageChoice::Spoken(Language::En)));
        assert_eq!(all.len(), Language::VARIANTS.len());
    }

    #[test]
    fn search_matches_word_starts_and_codes() {
        assert_eq!(options("fre"), vec![LanguageChoice::Spoken(Language::Fr)]);
        assert_eq!(
            options("CREOLE"),
            vec![LanguageChoice::Spoken(Language::Ht)]
        );
        assert_eq!(
            options("de"),
            vec![
                LanguageChoice::Automatic,
                LanguageChoice::Spoken(Language::De)
            ]
        );
        assert_eq!(options("zzz"), Vec::<LanguageChoice>::new());
    }
}
