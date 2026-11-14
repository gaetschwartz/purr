//! How Whisper spells languages, in whisper.cpp and in the browser's Whisper alike.

use super::tag::LanguageTag;

/// Tags whose primary language Whisper spells differently.
const SPELLED_DIFFERENTLY: &[(&str, &str)] = &[("jv", "jw")];

/// The code Whisper knows `tag` by; Whisper has no regional variants, so only the language counts.
#[must_use]
pub fn code(tag: &LanguageTag) -> &str {
    let primary = tag.primary();
    SPELLED_DIFFERENTLY
        .iter()
        .find(|(language, _)| *language == primary)
        .map_or(primary, |&(_, spelling)| spelling)
}

/// The tag for a code Whisper reports.
pub fn tag(code: &str) -> Result<LanguageTag, super::tag::LanguageTagError> {
    LanguageTag::parse(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(text: &str) -> LanguageTag {
        text.parse().unwrap()
    }

    #[test]
    fn javanese_is_jw_to_whisper_and_hebrew_is_he() {
        assert_eq!(code(&parsed("jv")), "jw");
        assert_eq!(code(&parsed("he")), "he");
        assert_eq!(code(&parsed("iw")), "he");
    }

    #[test]
    fn a_region_is_dropped() {
        assert_eq!(code(&parsed("pt-BR")), "pt");
    }

    #[test]
    fn what_whisper_reports_becomes_the_tag() {
        assert_eq!(tag("jw"), Ok(parsed("jv")));
        assert_eq!(tag("he"), Ok(parsed("he")));
        assert_eq!(tag("en"), Ok(parsed("en")));
    }
}
