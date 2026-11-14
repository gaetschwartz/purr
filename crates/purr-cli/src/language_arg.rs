use purr_common::language::{choice::LanguageChoice, known, tag::LanguageTag};

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

#[cfg(test)]
mod tests {
    use super::*;

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
