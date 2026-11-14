use purr_common::language::{known, tag::LanguageTag};

/// A readable name for a stored language code, the code itself when it is not known.
#[must_use]
pub fn language_label(code: &str) -> String {
    LanguageTag::parse(code)
        .ok()
        .and_then(|tag| known::display_name(&tag))
        .unwrap_or_else(|| code.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_codes_become_names() {
        assert_eq!(language_label("fr"), "French");
        assert_eq!(language_label("en"), "English");
    }

    #[test]
    fn codes_stored_under_an_old_spelling_become_names() {
        assert_eq!(language_label("iw"), "Hebrew");
        assert_eq!(language_label("jw"), "Javanese");
        assert_eq!(language_label("FR"), "French");
    }

    #[test]
    fn unknown_codes_are_kept() {
        assert_eq!(language_label("zz"), "zz");
        assert_eq!(language_label("not a code"), "not a code");
    }
}
