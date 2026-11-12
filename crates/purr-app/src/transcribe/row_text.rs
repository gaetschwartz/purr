//! A row's text cut into pieces by its words and by the search matches inside it.

use purr_common::word::Word;
use std::ops::Range;

/// A word placed in a row's trimmed text.
#[derive(Debug, Clone, PartialEq)]
pub struct WordSpan {
    /// The word's position in the segment's word list
    pub index: usize,
    pub range: Range<usize>,
    /// Where the word starts in the audio, in seconds, when known
    pub start: Option<f64>,
}

/// The words of a segment whose text is `text`, placed in `text.trim()`.
///
/// Word ranges index the segment's text as stored, while rows show it trimmed. A word that does
/// not lie inside the trimmed text on a character boundary, or that starts before the word
/// ahead of it ends, is left out.
#[must_use]
pub fn word_spans(text: &str, words: &[Word]) -> Vec<WordSpan> {
    let trimmed = text.trim();
    let lead = text.len() - text.trim_start().len();
    let mut spans: Vec<WordSpan> = Vec::new();
    let mut floor = 0;
    for (index, word) in words.iter().enumerate() {
        let (Some(start), Some(end)) = (
            (word.range.start as usize).checked_sub(lead),
            (word.range.end as usize).checked_sub(lead),
        ) else {
            continue;
        };
        if start < end && start >= floor && trimmed.get(start..end).is_some() {
            spans.push(WordSpan {
                index,
                range: start..end,
                start: word.start_seconds(),
            });
            floor = end;
        }
    }
    spans
}

/// What a piece belongs to when it lies inside a word.
#[derive(Debug, Clone, PartialEq)]
pub struct WordInfo {
    pub unsure: bool,
    pub start: Option<f64>,
    /// The piece ends the word
    pub last: bool,
}

/// A stretch of a row's text.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece<'a> {
    pub text: &'a str,
    pub word: Option<WordInfo>,
    /// The search match the piece is part of, numbered among all the matches of the transcript
    pub found: Option<usize>,
}

/// Cuts `text` at every word and match boundary, numbering the matches from `first_match`.
///
/// `unsure` lists the indices of the segment's words to flag; `matches` are sorted and do not
/// overlap, as `search::find` returns them. A match across a word boundary becomes several
/// pieces that share one number.
#[must_use]
pub fn pieces<'a>(
    text: &'a str,
    words: &[WordSpan],
    unsure: &[usize],
    matches: &[Range<usize>],
    first_match: usize,
) -> Vec<Piece<'a>> {
    let mut cuts: Vec<usize> = [0, text.len()]
        .into_iter()
        .chain(
            words
                .iter()
                .flat_map(|word| [word.range.start, word.range.end]),
        )
        .chain(matches.iter().flat_map(|found| [found.start, found.end]))
        .filter(|cut| text.is_char_boundary(*cut))
        .collect();
    cuts.sort_unstable();
    cuts.dedup();
    cuts.windows(2)
        .filter_map(|pair| {
            let (from, to) = (*pair.first()?, *pair.get(1)?);
            let slice = text.get(from..to)?;
            let word = words
                .iter()
                .find(|word| word.range.start <= from && to <= word.range.end)
                .map(|word| WordInfo {
                    unsure: unsure.contains(&word.index),
                    start: word.start,
                    last: to == word.range.end,
                });
            let found = matches
                .iter()
                .position(|found| found.start <= from && to <= found.end)
                .map(|position| first_match + position);
            Some(Piece {
                text: slice,
                word,
                found,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(start: u32, end: u32, start_cs: Option<u32>) -> Word {
        Word {
            range: start..end,
            start_cs,
            end_cs: None,
            score: Some(200),
        }
    }

    fn plain(text: &str) -> Piece<'_> {
        Piece {
            text,
            word: None,
            found: None,
        }
    }

    #[test]
    fn word_ranges_move_to_the_trimmed_text() {
        let text = " Et un grand ";
        let words = [
            word(1, 3, Some(10)),
            word(4, 6, None),
            word(7, 12, Some(90)),
        ];
        assert_eq!(
            word_spans(text, &words),
            vec![
                WordSpan {
                    index: 0,
                    range: 0..2,
                    start: Some(0.1)
                },
                WordSpan {
                    index: 1,
                    range: 3..5,
                    start: None
                },
                WordSpan {
                    index: 2,
                    range: 6..11,
                    start: Some(0.9)
                },
            ]
        );
    }

    #[test]
    fn words_outside_the_text_or_inside_a_character_are_left_out() {
        let text = " été fini";
        let words = [
            word(0, 1, None),
            word(2, 3, None),
            word(1, 4, None),
            word(6, 20, None),
            word(7, 11, None),
        ];
        let spans = word_spans(text, &words);
        assert_eq!(
            spans
                .iter()
                .map(|span| (span.index, span.range.clone()))
                .collect::<Vec<_>>(),
            vec![(2, 0..3), (4, 6..10)]
        );
    }

    #[test]
    fn overlapping_and_unordered_words_are_left_out() {
        let words = [word(4, 7, None), word(0, 3, None), word(5, 9, None)];
        let spans = word_spans("abc def ghi", &words);
        assert_eq!(
            spans.iter().map(|span| span.index).collect::<Vec<_>>(),
            vec![0]
        );
    }

    #[test]
    fn text_without_words_or_matches_is_one_piece() {
        assert_eq!(pieces("a cat", &[], &[], &[], 0), vec![plain("a cat")]);
        assert_eq!(pieces("", &[], &[], &[], 0), Vec::<Piece<'_>>::new());
    }

    #[test]
    fn matches_split_plain_text_and_are_numbered() {
        assert_eq!(
            pieces("a cat sat", &[], &[], &[Range { start: 2, end: 5 }], 4),
            vec![
                plain("a "),
                Piece {
                    text: "cat",
                    word: None,
                    found: Some(4)
                },
                plain(" sat"),
            ]
        );
    }

    #[test]
    fn words_flag_unsure_ones_and_mark_where_each_ends() {
        let spans = [
            WordSpan {
                index: 0,
                range: 0..1,
                start: Some(0.5),
            },
            WordSpan {
                index: 1,
                range: 2..5,
                start: None,
            },
        ];
        let info = |unsure, start, last| {
            Some(WordInfo {
                unsure,
                start,
                last,
            })
        };
        assert_eq!(
            pieces("a cat", &spans, &[1], &[], 0),
            vec![
                Piece {
                    text: "a",
                    word: info(false, Some(0.5), true),
                    found: None
                },
                plain(" "),
                Piece {
                    text: "cat",
                    word: info(true, None, true),
                    found: None
                },
            ]
        );
    }

    #[test]
    fn a_match_across_a_word_boundary_keeps_one_number_and_the_words() {
        let spans = [
            WordSpan {
                index: 0,
                range: 0..3,
                start: Some(1.0),
            },
            WordSpan {
                index: 1,
                range: 4..7,
                start: Some(2.0),
            },
        ];
        let found: Vec<(&str, Option<usize>, bool)> =
            pieces("big cat ran", &spans, &[], &[Range { start: 2, end: 6 }], 7)
                .iter()
                .map(|piece| (piece.text, piece.found, piece.word.is_some()))
                .collect();
        assert_eq!(
            found,
            vec![
                ("bi", None, true),
                ("g", Some(7), true),
                (" ", Some(7), false),
                ("ca", Some(7), true),
                ("t", None, true),
                (" ran", None, false),
            ]
        );
    }

    #[test]
    fn a_cut_inside_a_character_is_ignored() {
        let spans = [WordSpan {
            index: 0,
            range: 0..1,
            start: None,
        }];
        assert_eq!(pieces("été", &spans, &[], &[], 0), vec![plain("été")]);
    }
}
