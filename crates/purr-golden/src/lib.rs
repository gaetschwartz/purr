//! Comparison of a transcript with a recorded one that tolerates what differs between CPUs.
//!
//! The same model on another instruction set or compiler moves commas, segment boundaries and
//! timings, and on noisy audio an occasional word. A golden is therefore matched on its words,
//! with a word error rate allowed per case.

use serde::{Deserialize, Serialize};
use std::{
    fmt,
    path::{Path, PathBuf},
};

/// What a golden file holds: the transcript and, for a whole-source run, the audio length.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recording {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_duration: Option<f32>,
}

#[derive(Debug)]
pub enum GoldenError {
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
}

impl fmt::Display for GoldenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, .. } => write!(
                f,
                "cannot read the golden {}: record it with PURR_GOLDEN=update",
                path.display()
            ),
            Self::Write { path, .. } => write!(f, "cannot write the golden {}", path.display()),
            Self::Parse { path, .. } => write!(f, "the golden {} is not valid", path.display()),
        }
    }
}

impl std::error::Error for GoldenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } | Self::Write { source, .. } => Some(source),
            Self::Parse { source, .. } => Some(source),
        }
    }
}

impl Recording {
    pub fn read(path: &Path) -> Result<Self, GoldenError> {
        let contents = std::fs::read_to_string(path).map_err(|source| GoldenError::Read {
            path: path.to_owned(),
            source,
        })?;
        serde_json::from_str(&contents).map_err(|source| GoldenError::Parse {
            path: path.to_owned(),
            source,
        })
    }

    fn write(&self, path: &Path) -> Result<(), GoldenError> {
        let contents = serde_json::to_string_pretty(self).map_err(|source| GoldenError::Parse {
            path: path.to_owned(),
            source,
        })?;
        std::fs::write(path, contents + "\n").map_err(|source| GoldenError::Write {
            path: path.to_owned(),
            source,
        })
    }
}

/// Whether `PURR_GOLDEN=update` asks for the goldens to be recorded again.
pub fn updating() -> bool {
    std::env::var("PURR_GOLDEN").is_ok_and(|mode| mode == "update")
}

/// Compares `actual` with the golden at `path`, allowing `max_word_error_rate` on the words;
/// `PURR_GOLDEN=update` records `actual` there instead.
pub fn check(path: &Path, actual: &Recording, max_word_error_rate: f64) -> Result<(), GoldenError> {
    if updating() {
        return actual.write(path);
    }
    let golden = Recording::read(path)?;
    assert_text_close(&golden.text, &actual.text, max_word_error_rate);
    if let (Some(expected), Some(got)) = (golden.audio_duration, actual.audio_duration) {
        assert!(
            (expected - got).abs() < 0.05,
            "audio duration {got} differs from the golden {expected}"
        );
    }
    Ok(())
}

/// The lower-cased words of `text`: letters, digits and apostrophes inside a word; everything
/// else separates words.
pub fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .map(|word| word.trim_matches('\'').to_lowercase())
        .filter(|word| !word.is_empty())
        .collect()
}

/// Number of word insertions, deletions and substitutions that turn `reference` into `actual`.
fn edit_distance(reference: &[String], actual: &[String]) -> usize {
    let mut previous: Vec<usize> = (0..=actual.len()).collect();
    for (row, expected) in reference.iter().enumerate() {
        let mut left = row + 1;
        let mut current = vec![left];
        for (got, pair) in actual.iter().zip(previous.windows(2)) {
            let &[diagonal, above] = pair else { continue };
            left = (diagonal + usize::from(expected != got))
                .min(above + 1)
                .min(left + 1);
            current.push(left);
        }
        previous = current;
    }
    previous.last().copied().unwrap_or(0)
}

/// Word-level edit distance between `reference` and `actual`, divided by the length of
/// `reference`; 0 when the two have the same words.
#[allow(clippy::cast_precision_loss, reason = "word counts are far below 2^52")]
pub fn word_error_rate(reference: &str, actual: &str) -> f64 {
    let reference = words(reference);
    let actual = words(actual);
    if reference.is_empty() {
        return if actual.is_empty() { 0.0 } else { 1.0 };
    }
    edit_distance(&reference, &actual) as f64 / reference.len() as f64
}

/// Fails when the words of `actual` are further than `max_word_error_rate` from `reference`.
pub fn assert_text_close(reference: &str, actual: &str, max_word_error_rate: f64) {
    let rate = word_error_rate(reference, actual);
    assert!(
        rate <= max_word_error_rate,
        "word error rate {rate:.4} is above {max_word_error_rate:.4}\nreference: {reference}\nactual:    {actual}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn distance(reference: &str, actual: &str) -> usize {
        edit_distance(&words(reference), &words(actual))
    }

    #[test]
    fn punctuation_and_case_are_ignored() {
        assert_eq!(
            words("And so, my fellow Americans — don't ask."),
            ["and", "so", "my", "fellow", "americans", "don't", "ask"]
        );
    }

    #[test]
    fn identical_words_have_no_distance() {
        assert_eq!(distance("And so, my fellow", "and so my fellow."), 0);
    }

    #[test]
    fn a_substituted_word_counts_once() {
        assert_eq!(distance("one two three four", "one two tree four"), 1);
    }

    #[test]
    fn an_inserted_and_a_deleted_word_each_count_once() {
        assert_eq!(distance("a b c d", "a b x c d"), 1);
        assert_eq!(distance("a b c d", "a b d"), 1);
    }

    #[test]
    fn an_empty_side_costs_the_other_sides_length() {
        assert_eq!(distance("", "one two"), 2);
        assert_eq!(distance("one two", ""), 2);
    }

    #[test]
    fn the_rate_is_the_distance_over_the_reference_length() {
        assert!((word_error_rate("one two three four", "one two tree four") - 0.25).abs() < 1e-9);
        assert!(word_error_rate("", "").abs() < 1e-9);
        assert!((word_error_rate("", "word") - 1.0).abs() < 1e-9);
    }
}
