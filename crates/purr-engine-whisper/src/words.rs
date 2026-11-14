//! Words rebuilt from the tokens of one decoded segment.
//!
//! A token is a piece of a word. A token whose text starts with whitespace starts a new word, and
//! so does a token that starts a character of a script written without spaces (Chinese,
//! Japanese, Thai, Lao, Khmer, Burmese, Tibetan), or follows one. Tokens that only hold
//! punctuation or symbols stay with the word before them and do not count towards its score.
//! Ids at or above end-of-text are special or timestamp tokens and hold no text.

use purr_common::word::Word;

/// One token of a decoded segment.
#[derive(Debug, Clone, Copy)]
pub struct Token<'a> {
    pub id: i32,
    /// The token's text, which may hold only part of a UTF-8 character
    pub bytes: &'a [u8],
    /// Probability the decoder gave the token
    pub p: f32,
    /// Token start and end in centiseconds from the window start, negative when not computed
    pub t0: i64,
    pub t1: i64,
}

/// The words of a segment's trimmed text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentWords {
    pub text: String,
    pub words: Vec<Word>,
}

struct Building {
    start: usize,
    end: usize,
    min_p: Option<f32>,
    t0: i64,
    t1: i64,
}

/// Splits `raw`, the untrimmed text of a segment spanning `bounds` centiseconds of its window,
/// into words with the scores and times of `tokens`.
///
/// Times stay relative to the window. Returns `None` when the text tokens do not spell `raw`
/// exactly, so word ranges can never point at the wrong bytes.
#[must_use]
pub fn group(
    raw: &str,
    tokens: &[Token<'_>],
    eot: i32,
    bounds: (i64, i64),
) -> Option<SegmentWords> {
    let text_tokens: Vec<&Token<'_>> = tokens
        .iter()
        .filter(|token| token.id < eot && !token.bytes.is_empty())
        .collect();
    let spelled: Vec<u8> = text_tokens
        .iter()
        .flat_map(|token| token.bytes.iter().copied())
        .collect();
    if spelled != raw.as_bytes() {
        return None;
    }

    let mut built: Vec<Building> = Vec::new();
    let mut offset = 0;
    for token in text_tokens {
        let start = offset;
        let end = offset + token.bytes.len();
        offset = end;

        let first = raw.get(start..).and_then(|rest| rest.chars().next());
        let spaced = first.is_some_and(char::is_whitespace);
        let content = raw
            .get(start..end)
            .is_none_or(|piece| piece.chars().any(char::is_alphanumeric));
        let after_spaceless = built
            .last()
            .and_then(|word| raw.get(word.start..word.end))
            .and_then(|before| before.chars().rev().find(|c| c.is_alphanumeric()))
            .is_some_and(spaceless);
        let starts_word = built.is_empty()
            || spaced
            || first.is_some_and(|c| c.is_alphanumeric() && (spaceless(c) || after_spaceless));

        let leading = raw
            .get(start..)
            .map_or(0, |rest| rest.len() - rest.trim_start().len())
            .min(end - start);
        let p = content.then_some(token.p);
        match built.last_mut() {
            Some(word) if !starts_word => {
                word.end = end;
                word.min_p = min(word.min_p, p);
                word.t1 = token.t1;
            }
            Some(_) | None => built.push(Building {
                start: start + leading,
                end,
                min_p: p,
                t0: token.t0,
                t1: token.t1,
            }),
        }
    }

    let lead = raw.len() - raw.trim_start().len();
    let trimmed = raw.trim();
    let time = |t: i64| {
        (t >= 0)
            .then(|| t.clamp(bounds.0.max(0), bounds.1.max(bounds.0).max(0)))
            .and_then(|t| u32::try_from(t).ok())
    };
    let words = built
        .into_iter()
        .filter_map(|word| {
            let start = word.start.checked_sub(lead)?;
            let end = word.end.saturating_sub(lead).min(trimmed.len());
            let start_cs = time(word.t0);
            let end_cs = time(word.t1).map(|end| start_cs.map_or(end, |start| end.max(start)));
            (start < end).then_some(Word {
                range: u32::try_from(start).ok()?..u32::try_from(end).ok()?,
                start_cs,
                end_cs,
                score: word.min_p.map(Word::score_of),
            })
        })
        .collect();
    Some(SegmentWords {
        text: trimmed.to_owned(),
        words,
    })
}

fn min(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(p), None) | (None, Some(p)) => Some(p),
        (None, None) => None,
    }
}

/// Whether `c` belongs to a script that does not separate words with spaces.
fn spaceless(c: char) -> bool {
    matches!(
        u32::from(c),
        0x0E00..=0x0EFF   // Thai, Lao
            | 0x0F00..=0x0FFF // Tibetan
            | 0x1000..=0x109F // Myanmar
            | 0x1780..=0x17FF // Khmer
            | 0x3040..=0x30FF // Hiragana, Katakana
            | 0x31F0..=0x31FF // Katakana phonetic extensions
            | 0x3400..=0x4DBF // CJK extension A
            | 0x4E00..=0x9FFF // CJK unified ideographs
            | 0xF900..=0xFAFF // CJK compatibility ideographs
            | 0xFF66..=0xFF9F // Halfwidth katakana
            | 0x20000..=0x3134F // CJK extensions B to G
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOT: i32 = 50_257;
    const BEG: i32 = 50_364;

    fn token(id: i32, bytes: &[u8], p: f32, t0: i64, t1: i64) -> Token<'_> {
        Token {
            id,
            bytes,
            p,
            t0,
            t1,
        }
    }

    fn text(id: i32, text: &str, p: f32) -> Token<'_> {
        token(id, text.as_bytes(), p, -1, -1)
    }

    fn word(range: Range, score: Option<u8>) -> Word {
        Word {
            range,
            start_cs: None,
            end_cs: None,
            score,
        }
    }

    type Range = std::ops::Range<u32>;

    #[test]
    fn a_leading_space_starts_a_word_and_sub_word_pieces_join_it() {
        let tokens = [
            text(BEG, "[_BEG_]", 0.9),
            text(1, " And", 0.674),
            text(2, " micro", 0.4),
            text(3, "-", 0.9),
            text(4, "mach", 0.2),
            text(5, "ining", 0.9),
            text(BEG + 200, "[_TT_200]", 0.5),
        ];
        assert_eq!(
            group(" And micro-machining", &tokens, EOT, (0, 200)),
            Some(SegmentWords {
                text: "And micro-machining".to_owned(),
                words: vec![
                    word(0..3, Some(Word::score_of(0.674))),
                    word(4..19, Some(Word::score_of(0.2))),
                ],
            })
        );
    }

    #[test]
    fn punctuation_stays_with_its_word_and_out_of_its_score() {
        let tokens = [
            text(1, " Americans", 0.921),
            text(2, ",", 0.349),
            text(3, " ask", 0.749),
            text(4, ".", 0.1),
        ];
        assert_eq!(
            group(" Americans, ask.", &tokens, EOT, (0, 0)).map(|words| words.words),
            Some(vec![
                word(0..10, Some(Word::score_of(0.921))),
                word(11..15, Some(Word::score_of(0.749))),
            ])
        );
    }

    #[test]
    fn a_word_of_punctuation_alone_has_no_score() {
        let tokens = [
            text(1, " so", 0.9),
            text(2, " -", 0.3),
            text(3, " yes", 0.8),
        ];
        assert_eq!(
            group(" so - yes", &tokens, EOT, (0, 0)).map(|words| words.words),
            Some(vec![
                word(0..2, Some(Word::score_of(0.9))),
                word(3..4, None),
                word(5..8, Some(Word::score_of(0.8))),
            ])
        );
    }

    #[test]
    fn a_character_split_across_tokens_stays_in_one_word() {
        let cafe = " café".as_bytes();
        let (head, tail) = cafe.split_at(cafe.len() - 1);
        let tokens = [
            token(1, head, 0.8, -1, -1),
            token(2, tail, 0.6, -1, -1),
            text(3, " noir", 0.9),
        ];
        assert_eq!(
            group(" café noir", &tokens, EOT, (0, 0)),
            Some(SegmentWords {
                text: "café noir".to_owned(),
                words: vec![
                    word(0..5, Some(Word::score_of(0.6))),
                    word(6..10, Some(Word::score_of(0.9))),
                ],
            })
        );
    }

    #[test]
    fn each_token_of_a_spaceless_script_is_a_word() {
        let ni = "你".as_bytes();
        let (head, tail) = ni.split_at(1);
        let tokens = [
            token(1, head, 0.5, -1, -1),
            token(2, tail, 0.7, -1, -1),
            text(3, "好", 0.9),
            text(4, "。", 0.2),
            text(5, "OK", 0.6),
        ];
        assert_eq!(
            group("你好。OK", &tokens, EOT, (0, 0)).map(|words| words.words),
            Some(vec![
                word(0..3, Some(Word::score_of(0.5))),
                word(3..9, Some(Word::score_of(0.9))),
                word(9..11, Some(Word::score_of(0.6))),
            ])
        );
    }

    #[test]
    fn token_times_become_word_times_inside_the_segment() {
        let tokens = [
            token(1, b" ask", 0.7, 349, 424),
            token(2, b" not", 0.8, 424, 430),
            token(3, b",", 0.6, 430, 990),
        ];
        assert_eq!(
            group(" ask not,", &tokens, EOT, (340, 600)).map(|words| words.words),
            Some(vec![
                Word {
                    range: 0..3,
                    start_cs: Some(349),
                    end_cs: Some(424),
                    score: Some(Word::score_of(0.7)),
                },
                Word {
                    range: 4..8,
                    start_cs: Some(424),
                    end_cs: Some(600),
                    score: Some(Word::score_of(0.8)),
                },
            ])
        );
    }

    #[test]
    fn tokens_that_do_not_spell_the_text_give_no_words() {
        let tokens = [text(1, " hello", 0.9)];
        assert_eq!(group(" hullo", &tokens, EOT, (0, 0)), None);
    }

    #[test]
    fn an_empty_segment_has_no_words() {
        assert_eq!(
            group("", &[text(BEG, "[_BEG_]", 1.0)], EOT, (0, 0)),
            Some(SegmentWords {
                text: String::new(),
                words: Vec::new(),
            })
        );
    }
}
