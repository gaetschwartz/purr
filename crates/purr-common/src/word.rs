//! A word of a transcript segment, with the recogniser's timing and certainty.

use serde::{Deserialize, Serialize};
use std::ops::Range;

/// One word of a segment's text.
///
/// Stored as a five-element array (`[start, end, start_cs, end_cs, score]`) to keep saved
/// transcripts small.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "StoredWord", into = "StoredWord")]
pub struct Word {
    /// The word's bytes in its segment's text, without the space before it
    pub range: Range<u32>,
    /// Where the word starts on the source timeline, in centiseconds
    pub start_cs: Option<u32>,
    /// Where the word ends on the source timeline, in centiseconds
    pub end_cs: Option<u32>,
    /// The probability of the word's least certain token, from 0 (0.0) to 255 (1.0)
    pub score: Option<u8>,
}

#[derive(Serialize, Deserialize)]
struct StoredWord(u32, u32, Option<u32>, Option<u32>, Option<u8>);

impl From<StoredWord> for Word {
    fn from(StoredWord(start, end, start_cs, end_cs, score): StoredWord) -> Self {
        Self {
            range: start..end,
            start_cs,
            end_cs,
            score,
        }
    }
}

impl From<Word> for StoredWord {
    fn from(word: Word) -> Self {
        Self(
            word.range.start,
            word.range.end,
            word.start_cs,
            word.end_cs,
            word.score,
        )
    }
}

impl Word {
    /// The score of a token probability.
    #[must_use]
    pub fn score_of(probability: f32) -> u8 {
        (probability.clamp(0.0, 1.0) * 255.0).round() as u8
    }

    /// The word as it reads in `segment_text`, when its range lies inside it.
    #[must_use]
    pub fn text<'a>(&self, segment_text: &'a str) -> Option<&'a str> {
        segment_text.get(self.range.start as usize..self.range.end as usize)
    }

    /// Where the word starts, in seconds.
    #[must_use]
    pub fn start_seconds(&self) -> Option<f64> {
        self.start_cs.map(|cs| f64::from(cs) / 100.0)
    }

    /// Where the word ends, in seconds.
    #[must_use]
    pub fn end_seconds(&self) -> Option<f64> {
        self.end_cs.map(|cs| f64::from(cs) / 100.0)
    }

    /// The score as a probability between 0 and 1.
    #[must_use]
    pub fn probability(&self) -> Option<f64> {
        self.score.map(|score| f64::from(score) / 255.0)
    }

    /// The same word with its times moved later by `offset_cs`.
    #[must_use]
    pub fn delayed(self, offset_cs: u32) -> Self {
        Self {
            start_cs: self.start_cs.map(|cs| cs.saturating_add(offset_cs)),
            end_cs: self.end_cs.map(|cs| cs.saturating_add(offset_cs)),
            ..self
        }
    }

    /// The same word with its range moved later by `offset` bytes.
    #[must_use]
    pub fn shifted(self, offset: u32) -> Self {
        Self {
            range: self.range.start.saturating_add(offset)..self.range.end.saturating_add(offset),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word() -> Word {
        Word {
            range: 4..9,
            start_cs: Some(120),
            end_cs: None,
            score: Some(204),
        }
    }

    #[test]
    fn a_word_is_stored_as_one_short_array() -> serde_json::Result<()> {
        assert_eq!(serde_json::to_string(&word())?, "[4,9,120,null,204]");
        assert_eq!(serde_json::from_str::<Word>("[4,9,120,null,204]")?, word());
        Ok(())
    }

    #[test]
    fn the_text_is_read_from_the_segment() {
        assert_eq!(word().text("And Lucy, my"), Some("Lucy,"));
        assert_eq!(word().text("And"), None);
    }

    #[test]
    fn scores_span_the_byte() {
        assert_eq!(Word::score_of(0.0), 0);
        assert_eq!(Word::score_of(0.5), 128);
        assert_eq!(Word::score_of(1.0), 255);
        assert_eq!(Word::score_of(1.5), 255);
        assert_eq!(Word::score_of(-0.1), 0);
    }

    #[test]
    fn times_read_in_seconds() {
        assert_eq!(word().start_seconds(), Some(1.2));
        assert_eq!(word().end_seconds(), None);
        assert_eq!(word().probability(), Some(0.8));
    }

    #[test]
    fn moving_a_word_keeps_what_it_lacks() {
        assert_eq!(
            word().delayed(1_000).shifted(2),
            Word {
                range: 6..11,
                start_cs: Some(1_120),
                end_cs: None,
                score: Some(204),
            }
        );
    }
}
