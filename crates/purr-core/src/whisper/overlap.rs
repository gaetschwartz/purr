//! Joining the text of streaming windows whose audio overlaps.

/// Most words a match may span: more than the overlap can hold even in fast speech.
const MAX_MATCH_WORDS: usize = 6;

/// Words a window may decode around the shared words: the one cut in half at either edge.
const EDGE_SLACK: usize = 1;

/// Drops the words a window repeats from the end of the previous one.
///
/// Each window starts with audio the previous window already ended with, so both decode the
/// same words there. Words are compared case- and punctuation-insensitively; the text that is
/// kept is returned as decoded.
#[derive(Debug, Default)]
pub struct OverlapTrimmer {
    tail: Vec<String>,
}

impl OverlapTrimmer {
    /// The part of `text` that the previous window did not already transcribe, and remembers
    /// `text` as the previous window.
    pub fn trim<'a>(&mut self, text: &'a str) -> &'a str {
        let words = words(text);
        let kept = shared_prefix_end(&self.tail, &words)
            .map_or(text, |end| text[end..].trim_start_matches(is_separator));
        let first_tail = words.len().saturating_sub(MAX_MATCH_WORDS + EDGE_SLACK);
        self.tail = words
            .into_iter()
            .skip(first_tail)
            .map(|word| word.normalized)
            .collect();
        kept
    }

    /// Forgets the previous window, for a window that transcribed nothing.
    pub fn reset(&mut self) {
        self.tail.clear();
    }
}

struct Word {
    normalized: String,
    end: usize,
}

fn words(text: &str) -> Vec<Word> {
    let mut words = Vec::new();
    let mut start = None;
    for (index, c) in text.char_indices().chain([(text.len(), ' ')]) {
        match (start, c.is_whitespace()) {
            (None, false) => start = Some(index),
            (Some(begin), true) => {
                let normalized: String = text[begin..index]
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect();
                if !normalized.is_empty() {
                    words.push(Word {
                        normalized,
                        end: index,
                    });
                }
                start = None;
            }
            (None, true) | (Some(_), false) => {}
        }
    }
    words
}

/// Byte offset in the current window's text where its repeat of the previous window ends.
///
/// The shared run must end at most [`EDGE_SLACK`] words before the end of the previous window
/// and start at most [`EDGE_SLACK`] words into the current one. A single shared word only counts
/// when it is exactly the last and the first word, so common short words do not match by chance.
fn shared_prefix_end(previous: &[String], current: &[Word]) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for previous_end in (previous.len().saturating_sub(EDGE_SLACK + 1)..previous.len()).rev() {
        for current_start in 0..=EDGE_SLACK {
            for len in 1..=MAX_MATCH_WORDS.min(previous_end + 1) {
                let (Some(run), Some(previous_run)) = (
                    current.get(current_start..current_start + len),
                    previous.get(previous_end + 1 - len..=previous_end),
                ) else {
                    break;
                };
                let shared = previous_run
                    .iter()
                    .zip(run)
                    .all(|(previous, current)| *previous == current.normalized);
                let at_both_edges = previous_end + 1 == previous.len() && current_start == 0;
                let accepted = shared && (len >= 2 || at_both_edges);
                if accepted
                    && best.is_none_or(|(best_len, _)| len > best_len)
                    && let Some(last) = run.last()
                {
                    best = Some((len, last.end));
                }
            }
        }
    }
    best.map(|(_, end)| end)
}

fn is_separator(c: char) -> bool {
    c.is_whitespace() || matches!(c, ',' | '.' | ';' | ':' | '!' | '?' | '-')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trimmed(previous: &str, current: &str) -> String {
        let mut trimmer = OverlapTrimmer::default();
        trimmer.trim(previous);
        trimmer.trim(current).to_owned()
    }

    #[test]
    fn drops_the_words_both_windows_heard() {
        assert_eq!(
            trimmed(
                "the time has come for Americans to make important",
                "make important decisions about our nation's future"
            ),
            "decisions about our nation's future"
        );
    }

    #[test]
    fn ignores_case_and_punctuation_but_keeps_them_in_the_text() {
        assert_eq!(
            trimmed(
                "And so my fellow Americans ask",
                "Ask not what your country can do."
            ),
            "not what your country can do."
        );
    }

    #[test]
    fn tolerates_one_garbled_word_at_each_edge() {
        assert_eq!(
            trimmed(
                "This one has dramatic impact.",
                "and has dramatic details for effective decision"
            ),
            "details for effective decision"
        );
    }

    #[test]
    fn keeps_text_that_shares_nothing() {
        assert_eq!(
            trimmed("Roger me, beautiful.", "Okay, stand by there."),
            "Okay, stand by there."
        );
    }

    #[test]
    fn keeps_a_word_cut_differently_by_each_window() {
        assert_eq!(
            trimmed(
                "as the campaigns come to a close, republic",
                "Republicans, Democrats"
            ),
            "Republicans, Democrats"
        );
    }

    #[test]
    fn a_single_shared_word_must_sit_at_both_edges() {
        assert_eq!(
            trimmed("one of the best", "Most best of times"),
            "Most best of times"
        );
    }

    #[test]
    fn a_common_pair_deeper_into_the_window_is_not_an_overlap() {
        assert_eq!(
            trimmed("one of the best", "best. Most of the time"),
            "Most of the time"
        );
    }

    #[test]
    fn a_window_that_only_repeats_becomes_empty() {
        assert_eq!(trimmed("for your country.", "your country."), "");
    }

    #[test]
    fn a_phrase_longer_than_the_overlap_can_hold_is_kept() {
        assert_eq!(
            trimmed("a b c d e f g h i j", "a b c d e f g h i j k"),
            "a b c d e f g h i j k"
        );
    }

    #[test]
    fn the_first_window_is_kept_whole() {
        let mut trimmer = OverlapTrimmer::default();
        assert_eq!(trimmer.trim("And so my fellow"), "And so my fellow");
    }

    #[test]
    fn a_reset_forgets_the_previous_window() {
        let mut trimmer = OverlapTrimmer::default();
        trimmer.trim("ask what you can do for your country");
        trimmer.reset();
        assert_eq!(trimmer.trim("your country"), "your country");
    }

    #[test]
    fn compares_against_the_whole_previous_window_not_only_what_it_kept() {
        let mut trimmer = OverlapTrimmer::default();
        trimmer.trim("we have a problem here");
        assert_eq!(
            trimmer.trim("problem here. This is Houston"),
            "This is Houston"
        );
        assert_eq!(trimmer.trim("is Houston, say again"), "say again");
    }

    #[test]
    fn words_without_letters_or_digits_are_not_compared() {
        assert_eq!(
            trimmed("the cross head - screw", "head - screw design"),
            "design"
        );
    }
}
