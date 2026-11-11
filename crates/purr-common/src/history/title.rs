//! What a saved transcription is called, and the rules for renaming it.

/// The longest title, in characters.
pub const MAX_TITLE_CHARS: usize = 120;

const UNTITLED: &str = "Untitled transcript";

/// Why a typed title was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TitleProblem {
    #[error("Type a name.")]
    Empty,
    #[error("Use at most {MAX_TITLE_CHARS} characters.")]
    TooLong,
}

fn tidy(raw: &str) -> String {
    raw.split(char::is_whitespace)
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .filter(|letter| !letter.is_control())
        .collect()
}

/// A typed title, with whitespace tidied, when it is acceptable.
pub fn clean_title(raw: &str) -> Result<String, TitleProblem> {
    let title = tidy(raw);
    if title.is_empty() {
        Err(TitleProblem::Empty)
    } else if title.chars().count() > MAX_TITLE_CHARS {
        Err(TitleProblem::TooLong)
    } else {
        Ok(title)
    }
}

/// The title a new entry starts with: the file name, shortened if it is very long.
#[must_use]
pub fn default_title(file_name: &str) -> String {
    let title = tidy(file_name);
    if title.is_empty() {
        UNTITLED.to_owned()
    } else {
        title.chars().take(MAX_TITLE_CHARS).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_title_is_trimmed_and_collapsed() {
        assert_eq!(
            clean_title("  Réunion \t du   lundi\n"),
            Ok("Réunion du lundi".to_owned())
        );
    }

    #[test]
    fn a_blank_title_is_refused() {
        assert_eq!(clean_title(""), Err(TitleProblem::Empty));
        assert_eq!(clean_title(" \t\n "), Err(TitleProblem::Empty));
    }

    #[test]
    fn the_length_limit_counts_characters() {
        assert!(clean_title(&"é".repeat(MAX_TITLE_CHARS)).is_ok());
        assert_eq!(
            clean_title(&"é".repeat(MAX_TITLE_CHARS + 1)),
            Err(TitleProblem::TooLong)
        );
    }

    #[test]
    fn control_characters_are_dropped() {
        assert_eq!(clean_title("a\u{7}b"), Ok("ab".to_owned()));
    }

    #[test]
    fn a_new_entry_is_named_after_its_file() {
        assert_eq!(
            default_title("WhatsApp Ptt 2026-11-11.ogg"),
            "WhatsApp Ptt 2026-11-11.ogg"
        );
        assert_eq!(default_title(""), UNTITLED);
        assert_eq!(
            default_title(&"x".repeat(500)).chars().count(),
            MAX_TITLE_CHARS
        );
    }
}
