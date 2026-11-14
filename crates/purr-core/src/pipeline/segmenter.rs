//! Segments with honest spans, from whatever an engine recognised in a window.

use purr_common::word::Word;

use crate::engine::hypothesis::{Hypothesis, Segment, Segmentation, Span};

use super::policy::SAMPLE_RATE;

const SAMPLES_PER_CENTISECOND: usize = SAMPLE_RATE / 100;

/// A segment placed in its window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    /// In samples from the window start
    pub span: Span,
    pub text: String,
    /// Timed from the window start
    pub words: Vec<Word>,
}

/// Where a synthesized segment may end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segmenter {
    /// A silence between two words at least this long ends a segment, in centiseconds
    pub pause_cs: u32,
    /// A segment does not grow longer than this, in centiseconds
    pub max_cs: u32,
}

impl Default for Segmenter {
    fn default() -> Self {
        Self {
            pause_cs: 100,
            max_cs: 1_000,
        }
    }
}

/// Whether `word` ends a sentence: its last character, past closing quotes and brackets, is a
/// full stop, a question mark or an exclamation mark of a common script.
#[must_use]
pub fn ends_sentence(word: &str) -> bool {
    word.chars()
        .rev()
        .find(|c| {
            !matches!(
                c,
                '"' | '\'' | ')' | ']' | '»' | '”' | '’' | '」' | '』' | '）'
            )
        })
        .is_some_and(|c| {
            matches!(
                c,
                '.' | '?' | '!' | '…' | '。' | '？' | '！' | '؟' | '۔' | '।' | '॥' | '።'
            )
        })
}

impl Segmenter {
    /// The segments of `hypothesis` in a window of `len` samples.
    ///
    /// The engine's own segments are kept as they are; a segment without a span covers the
    /// window. Segments to synthesize are split at sentence ends, pauses and the length cap,
    /// and each ends with its last word, never past the window.
    #[must_use]
    pub fn parts(self, hypothesis: Hypothesis, len: usize) -> Vec<Part> {
        let whole = Span { start: 0, end: len };
        match hypothesis.segmentation {
            Segmentation::Native => hypothesis
                .segments
                .into_iter()
                .map(|segment| Part {
                    span: segment.span.unwrap_or(whole),
                    text: segment.text,
                    words: segment.words,
                })
                .collect(),
            Segmentation::Synthesize => hypothesis
                .segments
                .into_iter()
                .flat_map(|segment| self.split(segment, len))
                .collect(),
        }
    }

    fn split(self, segment: Segment, len: usize) -> Vec<Part> {
        let clamp = |span: Span| Span {
            start: span.start.min(len),
            end: span.end.min(len),
        };
        let times: Option<Vec<(u32, u32)>> = segment
            .words
            .iter()
            .map(|word| word.start_cs.zip(word.end_cs))
            .collect();
        let Some(times) = times.filter(|times| !times.is_empty()) else {
            return vec![Part {
                span: clamp(segment.span.unwrap_or(Span { start: 0, end: len })),
                text: segment.text,
                words: segment.words,
            }];
        };

        let mut starts = vec![0];
        let mut group_start = times.first().map_or(0, |(start, _)| *start);
        for (i, word) in segment.words.iter().enumerate() {
            let (Some(&(_, end)), Some(&(next_start, next_end))) = (times.get(i), times.get(i + 1))
            else {
                continue;
            };
            let sentence = word.text(&segment.text).is_some_and(ends_sentence);
            let pause = next_start.saturating_sub(end) >= self.pause_cs;
            let too_long = next_end.saturating_sub(group_start) > self.max_cs;
            if sentence || pause || too_long {
                starts.push(i + 1);
                group_start = next_start;
            }
        }

        let mut parts = Vec::with_capacity(starts.len());
        for (n, &first) in starts.iter().enumerate() {
            let last = starts.get(n + 1).map_or(segment.words.len(), |next| *next) - 1;
            let (Some(head), Some(&(start_cs, _)), Some(&(_, end_cs))) =
                (segment.words.get(first), times.get(first), times.get(last))
            else {
                continue;
            };
            let from = if n == 0 { 0 } else { head.range.start as usize };
            let to = starts
                .get(n + 1)
                .and_then(|next| segment.words.get(*next))
                .map_or(segment.text.len(), |next| next.range.start as usize);
            let Some(slice) = segment.text.get(from..to) else {
                continue;
            };
            let text = slice.trim().to_owned();
            let Ok(offset) = u32::try_from(from + slice.len() - slice.trim_start().len()) else {
                continue;
            };
            let words = segment
                .words
                .get(first..=last)
                .unwrap_or_default()
                .iter()
                .map(|word| Word {
                    range: word.range.start.saturating_sub(offset)
                        ..word.range.end.saturating_sub(offset),
                    ..word.clone()
                })
                .collect();
            parts.push(Part {
                span: clamp(Span {
                    start: start_cs as usize * SAMPLES_PER_CENTISECOND,
                    end: end_cs as usize * SAMPLES_PER_CENTISECOND,
                }),
                text,
                words,
            });
        }
        parts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::text::TextAssembler;

    const SECOND: usize = SAMPLE_RATE;

    fn timed(pieces: &[(&str, u32, u32)]) -> Segment {
        let mut text = TextAssembler::new();
        for (piece, start, end) in pieces {
            text.push(piece, Some(*start), Some(*end), None);
        }
        text.finish(None)
    }

    fn synthesized(segment: Segment) -> Hypothesis {
        Hypothesis {
            segments: vec![segment],
            language: None,
            segmentation: Segmentation::Synthesize,
        }
    }

    fn texts(parts: &[Part]) -> Vec<&str> {
        parts.iter().map(|part| part.text.as_str()).collect()
    }

    fn spans(parts: &[Part]) -> Vec<(usize, usize)> {
        parts
            .iter()
            .map(|part| (part.span.start, part.span.end))
            .collect()
    }

    #[test]
    fn sentences_end_segments_and_spans_end_with_their_last_word() {
        let parts = Segmenter::default().parts(
            synthesized(timed(&[
                ("Hello", 10, 50),
                (" there.", 50, 90),
                (" How", 100, 130),
                (" are", 130, 150),
                (" you?", 150, 190),
            ])),
            3 * SECOND,
        );
        assert_eq!(texts(&parts), ["Hello there.", "How are you?"]);
        assert_eq!(spans(&parts), [(1_600, 14_400), (16_000, 30_400)]);
    }

    #[test]
    fn multibyte_sentence_ends_split_and_word_ranges_follow_the_new_text() {
        let parts = Segmenter::default().parts(
            synthesized(timed(&[
                ("今日は", 0, 40),
                ("晴れ。", 40, 80),
                ("¿Qué", 90, 120),
                (" tal?", 120, 150),
                (" مرحبا؟", 160, 200),
            ])),
            3 * SECOND,
        );
        assert_eq!(texts(&parts), ["今日は晴れ。", "¿Qué tal?", "مرحبا؟"]);
        let words: Vec<Vec<Option<&str>>> = parts
            .iter()
            .map(|part| {
                part.words
                    .iter()
                    .map(|word| word.text(&part.text))
                    .collect()
            })
            .collect();
        assert_eq!(
            words,
            [
                vec![Some("今日は"), Some("晴れ。")],
                vec![Some("¿Qué"), Some("tal?")],
                vec![Some("مرحبا؟")]
            ]
        );
    }

    #[test]
    fn text_before_the_first_word_stays_with_it() {
        let mut segment = timed(&[("Hi.\"", 0, 40), (" Bye", 50, 90)]);
        segment.text.insert(0, '"');
        for word in &mut segment.words {
            word.range = word.range.start + 1..word.range.end + 1;
        }
        let parts = Segmenter::default().parts(synthesized(segment), 3 * SECOND);
        assert_eq!(texts(&parts), ["\"Hi.\"", "Bye"]);
        assert_eq!(
            parts
                .first()
                .and_then(|part| part.words.first().and_then(|word| word.text(&part.text))),
            Some("Hi.\"")
        );
    }

    #[test]
    fn a_long_pause_ends_a_segment() {
        let parts = Segmenter::default().parts(
            synthesized(timed(&[
                ("one", 0, 40),
                (" two", 50, 90),
                (" three", 200, 240),
            ])),
            3 * SECOND,
        );
        assert_eq!(texts(&parts), ["one two", "three"]);
    }

    #[test]
    fn a_segment_stops_growing_at_the_length_cap() {
        let pieces: Vec<(String, u32, u32)> = (0..30)
            .map(|i| (format!(" w{i}"), i * 50, i * 50 + 45))
            .collect();
        let borrowed: Vec<(&str, u32, u32)> = pieces
            .iter()
            .map(|(text, start, end)| (text.as_str(), *start, *end))
            .collect();
        let parts = Segmenter::default().parts(synthesized(timed(&borrowed)), 20 * SECOND);
        assert_eq!(parts.len(), 2);
        assert!(parts.iter().all(|part| {
            let words = &part.words;
            let first = words.first().and_then(|word| word.start_cs).unwrap_or(0);
            let last = words.last().and_then(|word| word.end_cs).unwrap_or(0);
            last - first <= 1_000
        }));
    }

    #[test]
    fn a_span_never_reaches_past_the_window() {
        let parts = Segmenter::default().parts(synthesized(timed(&[("late", 90, 130)])), SECOND);
        assert_eq!(spans(&parts), [(14_400, SECOND)]);
    }

    #[test]
    fn untimed_words_stay_one_segment_over_the_window() {
        let mut text = TextAssembler::new();
        text.push("No.", None, None, None);
        text.push(" Times.", None, None, None);
        let parts = Segmenter::default().parts(synthesized(text.finish(None)), 2 * SECOND);
        assert_eq!(texts(&parts), ["No. Times."]);
        assert_eq!(spans(&parts), [(0, 2 * SECOND)]);
    }

    #[test]
    fn native_segments_pass_through_untouched() {
        let segment = Segment {
            text: "Ask not. What your country".to_owned(),
            span: Some(Span {
                start: 0,
                end: 40 * SECOND,
            }),
            words: timed(&[("Ask", 0, 40), (" not.", 40, 80), (" What", 300, 340)]).words,
        };
        let parts = Segmenter::default().parts(
            Hypothesis {
                segments: vec![segment.clone()],
                language: None,
                segmentation: Segmentation::Native,
            },
            30 * SECOND,
        );
        assert_eq!(
            parts,
            [Part {
                span: Span {
                    start: 0,
                    end: 40 * SECOND
                },
                text: segment.text,
                words: segment.words,
            }]
        );
    }

    #[test]
    fn sentence_ends_are_found_past_closing_quotes() {
        assert!(ends_sentence("done.\""));
        assert!(ends_sentence("終わり。」"));
        assert!(ends_sentence("Wirklich?"));
        assert!(!ends_sentence("Mr"));
        assert!(!ends_sentence("well,"));
    }
}
