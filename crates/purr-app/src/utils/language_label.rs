use purr_common::Language;

/// Upper-cases the first letter of `word`.
#[must_use]
pub fn capitalized(word: &str) -> String {
    let mut chars = word.chars();
    chars
        .next()
        .map(|first| first.to_uppercase().chain(chars).collect())
        .unwrap_or_default()
}

/// A readable name for a Whisper language code, the code itself when it is not known.
#[must_use]
pub fn language_label(code: &str) -> String {
    code.parse::<Language>().map_or_else(
        |()| code.to_owned(),
        |language| capitalized(language.name()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capitalizes_the_first_letter_only() {
        assert_eq!(capitalized("french"), "French");
        assert_eq!(capitalized(""), "");
    }

    #[test]
    fn known_codes_become_names() {
        assert_eq!(language_label("fr"), "French");
        assert_eq!(language_label("en"), "English");
    }

    #[test]
    fn unknown_codes_are_kept() {
        assert_eq!(language_label("zz"), "zz");
    }
}
