//! The language codes a transcribe.cpp model lists, and the tags they stand for.

use purr_common::language::{support::LanguageSupport, tag::LanguageTag};

use crate::error::TranscribeEngineError;

/// The languages a model lists, each with the code the library knows it by.
///
/// The library compares a hint with its list as text, so a hint goes out spelled as the model's
/// own list spells it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLanguages {
    support: LanguageSupport,
    codes: Vec<(LanguageTag, String)>,
}

impl ModelLanguages {
    /// Reads the codes of `capabilities.languages`; a model that lists none is not tied to any.
    pub fn from_codes(codes: &[String]) -> Result<Self, TranscribeEngineError> {
        let codes = codes
            .iter()
            .map(|code| {
                LanguageTag::parse(code)
                    .map(|tag| (tag, code.clone()))
                    .map_err(|source| TranscribeEngineError::LanguageCode {
                        code: code.clone(),
                        source,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let support = if codes.is_empty() {
            LanguageSupport::Any
        } else {
            LanguageSupport::Only(codes.iter().map(|(tag, _)| tag.clone()).collect())
        };
        Ok(Self { support, codes })
    }

    #[must_use]
    pub fn support(&self) -> &LanguageSupport {
        &self.support
    }

    /// The code to give the library for a hint of `tag`; `None` when the model lists no
    /// languages, which leaves nothing to steer.
    #[must_use]
    pub fn code(&self, tag: &LanguageTag) -> Option<&str> {
        let listed = self.support.listed(tag)?;
        self.codes
            .iter()
            .find(|(candidate, _)| candidate == listed)
            .map(|(_, code)| code.as_str())
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    fn tag(text: &str) -> LanguageTag {
        text.parse().unwrap()
    }

    fn languages(codes: &[&str]) -> ModelLanguages {
        let codes: Vec<String> = codes.iter().map(|code| (*code).to_owned()).collect();
        ModelLanguages::from_codes(&codes).unwrap()
    }

    #[test]
    fn the_model_decides_which_languages_it_is_asked_in() {
        let languages = languages(&["de", "fr"]);
        assert!(languages.support().supports(&tag("de")));
        assert!(languages.support().supports(&tag("fr-CH")));
        assert!(!languages.support().supports(&tag("ja")));
    }

    #[test]
    fn a_regional_hint_goes_out_as_the_code_the_model_lists() {
        let languages = languages(&["yue", "fr", "en-US"]);
        assert_eq!(languages.code(&tag("fr-CH")), Some("fr"));
        assert_eq!(languages.code(&tag("yue")), Some("yue"));
        assert_eq!(languages.code(&tag("en")), Some("en-US"));
        assert_eq!(languages.code(&tag("ja")), None);
    }

    #[test]
    fn a_model_that_lists_no_language_takes_any_and_is_given_no_hint() {
        let languages = languages(&[]);
        assert_eq!(languages.support(), &LanguageSupport::Any);
        assert_eq!(languages.code(&tag("fr")), None);
    }

    #[test]
    fn a_language_code_that_is_no_tag_is_refused() {
        let codes = ["de".to_owned(), "<|x|>".to_owned()];
        assert!(matches!(
            ModelLanguages::from_codes(&codes),
            Err(TranscribeEngineError::LanguageCode { .. })
        ));
    }
}
