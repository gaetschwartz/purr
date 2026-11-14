//! Segment text built from words, for engines that report words rather than text.

use purr_common::word::Word;

use super::hypothesis::{Segment, Span};

/// Joins words into a segment's text, keeping each word's byte range in it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TextAssembler {
    text: String,
    words: Vec<Word>,
}

impl TextAssembler {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one word as the engine spelled it: whitespace before it separates it from the word
    /// before, and a piece without any joins straight on. A piece of whitespace alone is dropped.
    pub fn push(
        &mut self,
        piece: &str,
        start_cs: Option<u32>,
        end_cs: Option<u32>,
        score: Option<u8>,
    ) {
        let word = piece.trim();
        if word.is_empty() {
            return;
        }
        if !self.text.is_empty() && piece.starts_with(char::is_whitespace) {
            self.text.push(' ');
        }
        let start = self.text.len();
        self.text.push_str(word);
        let (Ok(start), Ok(end)) = (u32::try_from(start), u32::try_from(self.text.len())) else {
            return;
        };
        self.words.push(Word {
            range: start..end,
            start_cs,
            end_cs,
            score,
        });
    }

    /// The segment the words make up.
    #[must_use]
    pub fn finish(self, span: Option<Span>) -> Segment {
        Segment {
            text: self.text,
            span,
            words: self.words,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(range: std::ops::Range<u32>, start_cs: u32, end_cs: u32) -> Word {
        Word {
            range,
            start_cs: Some(start_cs),
            end_cs: Some(end_cs),
            score: None,
        }
    }

    #[test]
    fn words_are_joined_by_the_spaces_before_them() {
        let mut text = TextAssembler::new();
        text.push(" Grüß", Some(0), Some(40), None);
        text.push(" Gott", Some(40), Some(80), None);
        text.push("!", Some(80), Some(85), None);
        text.push("  ", None, None, None);
        assert_eq!(
            text.finish(None),
            Segment {
                text: "Grüß Gott!".to_owned(),
                span: None,
                words: vec![word(0..6, 0, 40), word(7..11, 40, 80), word(11..12, 80, 85)],
            }
        );
    }

    #[test]
    fn words_of_a_spaceless_script_join_without_spaces() {
        let mut text = TextAssembler::new();
        text.push("你好", Some(0), Some(30), None);
        text.push("。", Some(30), Some(35), None);
        let segment = text.finish(None);
        assert_eq!(segment.text, "你好。");
        assert_eq!(
            segment
                .words
                .iter()
                .map(|word| word.text(&segment.text))
                .collect::<Vec<_>>(),
            [Some("你好"), Some("。")]
        );
    }
}
