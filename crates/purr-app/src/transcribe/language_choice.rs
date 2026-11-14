//! The list the language picker offers.

use purr_common::language::{
    choice::LanguageChoice,
    known::{self, KnownLanguage},
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

    #[test]
    fn search_matches_word_starts_and_codes() {
        assert_eq!(options("fre"), vec![spoken("fr")]);
        assert_eq!(options("CREOLE"), vec![spoken("ht")]);
        assert_eq!(options("de"), vec![LanguageChoice::Automatic, spoken("de")]);
        assert_eq!(options("hebr"), vec![spoken("he")]);
        assert_eq!(options("zzz"), Vec::<LanguageChoice>::new());
    }
}
