use purr_common::{
    language::{choice::LanguageChoice, known, support::LanguageSupport, tag::LanguageTag},
    model::profile::{HintEffect, ModelProfile},
};
use std::{fmt, sync::Arc};

const EXAMPLES: &str = "a name like `french`, a code like `fr`, a tag like `pt-BR`, or `auto`";

/// Parses `--language`: a language by name or other name, a language tag, or `auto`.
pub fn parse_language(text: &str) -> Result<LanguageChoice, String> {
    let wanted = text.trim().to_lowercase();
    if wanted == "auto" || wanted == "automatic" {
        return Ok(LanguageChoice::Automatic);
    }
    if let Some(known) = known::find(&wanted) {
        return Ok(LanguageChoice::Spoken(known.tag.clone()));
    }
    LanguageTag::parse(&wanted)
        .map(LanguageChoice::Spoken)
        .map_err(|_| format!("`{text}` is not a language; use {EXAMPLES}"))
}

/// The languages a model transcribes, as the help of an error lists them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguageList(Arc<[LanguageTag]>);

impl fmt::Display for LanguageList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tags: Vec<&str> = self.0.iter().map(LanguageTag::as_str).collect();
        f.write_str(&tags.join(", "))
    }
}

/// Why `--language` does not fit the model.
#[oopsie::oopsie]
#[derive(Clone, PartialEq, Eq)]
pub enum LanguageProblem {
    #[oopsie(
        "{model} does not transcribe {language}",
        code = "purr::language::unsupported",
        help = "It transcribes: {supported}"
    )]
    Unsupported {
        model: String,
        language: LanguageTag,
        supported: LanguageList,
    },
    #[oopsie(
        "{model} cannot find the language itself",
        code = "purr::language::required",
        help = "Name the spoken language with --language"
    )]
    Required { model: String },
}

/// What `--language` does with the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageVerdict {
    Applies,
    /// The model finds the language itself and takes no hint
    HintIgnored,
}

/// Checks `--language` against what the model of `profile` can do, before the model is loaded.
pub fn check(
    choice: Option<&LanguageChoice>,
    model: &str,
    profile: &ModelProfile,
) -> Result<LanguageVerdict, LanguageProblem> {
    match choice {
        Some(LanguageChoice::Spoken(language)) => match &profile.languages {
            LanguageSupport::Only(tags) if !profile.languages.supports(language) => {
                Err(LanguageProblem::Unsupported {
                    model: model.to_owned(),
                    language: language.clone(),
                    supported: LanguageList(Arc::clone(tags)),
                })
            }
            LanguageSupport::Only(_) | LanguageSupport::Any => Ok(match profile.language_hint {
                HintEffect::Ignored => LanguageVerdict::HintIgnored,
                HintEffect::Steers | HintEffect::Required => LanguageVerdict::Applies,
            }),
        },
        Some(LanguageChoice::Automatic) | None => match profile.language_hint {
            HintEffect::Required if profile.languages.single().is_none() => {
                Err(LanguageProblem::Required {
                    model: model.to_owned(),
                })
            }
            HintEffect::Required | HintEffect::Steers | HintEffect::Ignored => {
                Ok(LanguageVerdict::Applies)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn few() -> LanguageList {
        LanguageList(Arc::from(
            ["en", "fr", "de"].map(|tag| tag.parse().unwrap()),
        ))
    }

    fn parakeet_like() -> ModelProfile {
        let mut profile = ModelProfile::browser(LanguageSupport::Only(
            ["en", "fr", "de"]
                .iter()
                .map(|tag| tag.parse().unwrap())
                .collect(),
        ));
        profile.language_hint = HintEffect::Ignored;
        profile
    }

    fn verdict(
        text: Option<&str>,
        profile: &ModelProfile,
    ) -> Result<LanguageVerdict, LanguageProblem> {
        let choice = text.map(|text| parse_language(text).unwrap());
        check(choice.as_ref(), "model", profile)
    }

    #[test]
    fn a_language_the_model_does_not_know_is_a_typed_error_listing_what_it_knows() {
        assert_eq!(
            verdict(Some("ja"), &parakeet_like()),
            Err(LanguageProblem::Unsupported {
                model: "model".to_owned(),
                language: "ja".parse().unwrap(),
                supported: few(),
            })
        );
    }

    #[test]
    fn a_known_language_is_only_a_warning_for_a_model_that_ignores_the_hint() {
        assert_eq!(
            verdict(Some("fr"), &parakeet_like()),
            Ok(LanguageVerdict::HintIgnored)
        );
        assert_eq!(
            verdict(None, &parakeet_like()),
            Ok(LanguageVerdict::Applies)
        );
        assert_eq!(
            verdict(Some("auto"), &parakeet_like()),
            Ok(LanguageVerdict::Applies)
        );
    }

    #[test]
    fn a_model_that_steers_takes_the_languages_it_knows_and_refuses_others() {
        let mut profile = parakeet_like();
        profile.language_hint = HintEffect::Steers;
        assert_eq!(
            verdict(Some("pt"), &profile),
            Err(LanguageProblem::Unsupported {
                model: "model".to_owned(),
                language: "pt".parse().unwrap(),
                supported: few(),
            })
        );
        assert_eq!(verdict(Some("de"), &profile), Ok(LanguageVerdict::Applies));
        let any = ModelProfile::browser(LanguageSupport::Any);
        assert_eq!(verdict(Some("ja"), &any), Ok(LanguageVerdict::Applies));
    }

    #[test]
    fn the_embedded_qwen3_asr_takes_the_hint_for_its_languages_by_name_or_code() {
        let registry = purr_core::model::registry::Registry::embedded().unwrap();
        let profile = &registry.get("qwen3-asr-0.6b").unwrap().profile;
        for wanted in ["fr", "French", "cantonese", "fil"] {
            assert_eq!(
                verdict(Some(wanted), profile),
                Ok(LanguageVerdict::Applies),
                "{wanted}"
            );
        }
        assert_eq!(parse_language("Cantonese"), spoken("yue"));
        assert!(matches!(
            verdict(Some("he"), profile),
            Err(LanguageProblem::Unsupported { .. })
        ));
        assert_eq!(verdict(None, profile), Ok(LanguageVerdict::Applies));
    }

    #[test]
    fn a_model_that_cannot_detect_needs_a_language() {
        let mut profile = ModelProfile::browser(LanguageSupport::Any);
        profile.language_hint = HintEffect::Required;
        assert_eq!(
            verdict(None, &profile),
            Err(LanguageProblem::Required {
                model: "model".to_owned()
            })
        );
        assert_eq!(verdict(Some("fr"), &profile), Ok(LanguageVerdict::Applies));
    }

    fn spoken(text: &str) -> Result<LanguageChoice, String> {
        Ok(LanguageChoice::Spoken(text.parse().unwrap()))
    }

    #[test]
    fn every_listed_language_is_found_by_its_code_and_its_name() {
        for listed in known::all() {
            assert_eq!(
                parse_language(listed.tag.as_str()),
                Ok(LanguageChoice::Spoken(listed.tag.clone()))
            );
            assert_eq!(
                parse_language(listed.name),
                Ok(LanguageChoice::Spoken(listed.tag.clone()))
            );
        }
    }

    #[test]
    fn other_names_and_old_codes_are_found() {
        assert_eq!(parse_language("Flemish"), spoken("nl"));
        assert_eq!(parse_language("castilian"), spoken("es"));
        assert_eq!(parse_language("iw"), spoken("he"));
        assert_eq!(parse_language("jw"), spoken("jv"));
    }

    #[test]
    fn any_valid_tag_is_accepted() {
        assert_eq!(parse_language("pt_br"), spoken("pt-BR"));
        assert_eq!(parse_language("yue"), spoken("yue"));
    }

    #[test]
    fn auto_asks_for_detection() {
        assert_eq!(parse_language("auto"), Ok(LanguageChoice::Automatic));
        assert_eq!(parse_language("Automatic"), Ok(LanguageChoice::Automatic));
    }

    #[test]
    fn something_that_is_no_language_is_refused_with_examples() {
        assert_eq!(
            parse_language("klingon"),
            Err("`klingon` is not a language; use a name like `french`, a code like `fr`, a tag like `pt-BR`, or `auto`".to_owned())
        );
    }
}
