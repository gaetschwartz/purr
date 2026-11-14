//! A transcribe.cpp transcript as a pipeline hypothesis.

use purr_common::{language::tag::LanguageTag, word::Word};
use purr_core::engine::{
    hypothesis::{Hypothesis, LanguageEstimate, Segmentation, Span},
    text::TextAssembler,
};
use transcribe_cpp::{Segment, TimestampKind, Token, Transcript};

use crate::error::TranscribeEngineError;

const SAMPLES_PER_MILLISECOND: i64 = 16;

/// What `transcript` recognised, with the language the library reports when `report_language`.
///
/// A model that times its words gives long segments for the pipeline to split from the word
/// times; one that gives no word times gives segments as it means them, with spans when it
/// times them and without when it does not.
pub fn hypothesis(
    transcript: &Transcript,
    report_language: bool,
) -> Result<Hypothesis, TranscribeEngineError> {
    let language = if report_language {
        estimate(transcript.language.as_deref())?
    } else {
        None
    };
    let segments = transcript
        .segments
        .iter()
        .map(|segment| purr_segment(transcript, segment))
        .filter(|segment| !segment.text.is_empty())
        .collect();
    Ok(Hypothesis {
        segments,
        language,
        segmentation: segmentation(transcript.timestamp_kind),
    })
}

fn segmentation(kind: TimestampKind) -> Segmentation {
    match kind {
        TimestampKind::Word | TimestampKind::Token => Segmentation::Synthesize,
        TimestampKind::None | TimestampKind::Auto | TimestampKind::Segment => Segmentation::Native,
    }
}

fn estimate(code: Option<&str>) -> Result<Option<LanguageEstimate>, TranscribeEngineError> {
    code.map(|code| {
        LanguageTag::parse(code)
            .map(|tag| LanguageEstimate {
                tag,
                confidence: None,
            })
            .map_err(|source| TranscribeEngineError::LanguageCode {
                code: code.to_owned(),
                source,
            })
    })
    .transpose()
}

fn purr_segment(
    transcript: &Transcript,
    segment: &Segment,
) -> purr_core::engine::hypothesis::Segment {
    let span = (segment.t1_ms > segment.t0_ms).then(|| Span {
        start: samples(segment.t0_ms),
        end: samples(segment.t1_ms),
    });
    let words = slice(&transcript.words, segment.first_word, segment.n_words);
    if words.is_empty() {
        return purr_core::engine::hypothesis::Segment {
            text: segment.text.trim().to_owned(),
            span,
            words: Vec::new(),
        };
    }
    let mut text = TextAssembler::new();
    for word in words {
        let tokens = slice(&transcript.tokens, word.first_token, word.n_tokens);
        text.push(
            &spelling(word, tokens),
            centiseconds(word.t0_ms),
            centiseconds(word.t1_ms),
            score(tokens),
        );
    }
    text.finish(span)
}

fn slice<T>(rows: &[T], first: i32, count: i32) -> &[T] {
    let (Ok(first), Ok(count)) = (usize::try_from(first), usize::try_from(count)) else {
        return &[];
    };
    rows.get(first..first.saturating_add(count)).unwrap_or(&[])
}

/// The word as its tokens spell it, which keeps the space before it; a word whose tokens the
/// transcript does not hold is separated from the one before by a space.
fn spelling(word: &transcribe_cpp::Word, tokens: &[Token]) -> String {
    if tokens.is_empty() {
        format!(" {}", word.text)
    } else {
        tokens.iter().map(|token| token.text.as_str()).collect()
    }
}

/// The probability of the least certain token, when the library gives one.
fn score(tokens: &[Token]) -> Option<u8> {
    tokens
        .iter()
        .map(|token| token.p)
        .filter(|probability| !probability.is_nan())
        .min_by(f32::total_cmp)
        .map(Word::score_of)
}

fn centiseconds(milliseconds: i64) -> Option<u32> {
    u32::try_from(milliseconds.max(0).saturating_add(5) / 10).ok()
}

fn samples(milliseconds: i64) -> usize {
    usize::try_from(milliseconds.max(0).saturating_mul(SAMPLES_PER_MILLISECOND)).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    fn word(
        text: &str,
        t0_ms: i64,
        t1_ms: i64,
        first_token: i32,
        n_tokens: i32,
    ) -> transcribe_cpp::Word {
        transcribe_cpp::Word {
            t0_ms,
            t1_ms,
            first_token,
            n_tokens,
            text: text.to_owned(),
            ..transcribe_cpp::Word::default()
        }
    }

    fn token(text: &str, p: f32) -> Token {
        Token {
            text: text.to_owned(),
            p,
            ..Token::default()
        }
    }

    fn segment(first_word: i32, n_words: i32, t0_ms: i64, t1_ms: i64, text: &str) -> Segment {
        Segment {
            t0_ms,
            t1_ms,
            first_word,
            n_words,
            text: text.to_owned(),
            ..Segment::default()
        }
    }

    fn so_my_friend() -> Transcript {
        Transcript {
            text: "So, my friend".to_owned(),
            timestamp_kind: TimestampKind::Token,
            segments: vec![segment(0, 3, 240, 1_760, "So, my friend")],
            words: vec![
                word("So,", 240, 1_040, 0, 2),
                word("my", 1_040, 1_280, 2, 1),
                word("friend", 1_280, 1_760, 3, 2),
            ],
            tokens: vec![
                token(" So", 0.9998),
                token(",", 0.5),
                token(" my", 0.9997),
                token(" fri", 0.2),
                token("end", f32::NAN),
            ],
            ..Transcript::default()
        }
    }

    fn only_segment(hypothesis: Hypothesis) -> purr_core::engine::hypothesis::Segment {
        let [segment] = <[_; 1]>::try_from(hypothesis.segments).expect("one segment");
        segment
    }

    #[test]
    fn words_keep_the_spaces_their_tokens_carry_with_times_in_centiseconds() {
        let segment = only_segment(hypothesis(&so_my_friend(), false).unwrap());
        assert_eq!(segment.text, "So, my friend");
        assert_eq!(
            segment.span,
            Some(Span {
                start: 3_840,
                end: 28_160
            })
        );
        let timed: Vec<_> = segment
            .words
            .iter()
            .map(|word| (word.text(&segment.text), word.start_cs, word.end_cs))
            .collect();
        assert_eq!(
            timed,
            [
                (Some("So,"), Some(24), Some(104)),
                (Some("my"), Some(104), Some(128)),
                (Some("friend"), Some(128), Some(176)),
            ]
        );
    }

    #[test]
    fn a_word_scores_as_its_least_certain_token() {
        let segment = only_segment(hypothesis(&so_my_friend(), false).unwrap());
        let scores: Vec<_> = segment.words.iter().map(|word| word.score).collect();
        assert_eq!(scores, [Some(128), Some(255), Some(51)]);
    }

    #[test]
    fn a_word_without_a_probability_has_no_score() {
        let mut transcript = so_my_friend();
        transcript.tokens = vec![
            token(" So", f32::NAN),
            token(",", f32::NAN),
            token(" my", 0.9),
            token(" fri", f32::NAN),
            token("end", f32::NAN),
        ];
        let segment = only_segment(hypothesis(&transcript, false).unwrap());
        let scores: Vec<_> = segment.words.iter().map(|word| word.score).collect();
        assert_eq!(scores, [None, Some(230), None]);
    }

    #[test]
    fn words_without_tokens_are_separated_by_spaces() {
        let mut transcript = so_my_friend();
        transcript.tokens.clear();
        let segment = only_segment(hypothesis(&transcript, false).unwrap());
        assert_eq!(segment.text, "So, my friend");
        assert_eq!(segment.words.len(), 3);
        assert_eq!(segment.words[0].score, None);
    }

    #[test]
    fn a_segment_without_words_keeps_its_trimmed_text() {
        let transcript = Transcript {
            segments: vec![segment(0, 0, 0, 0, "  Hello  ")],
            ..Transcript::default()
        };
        let segment = only_segment(hypothesis(&transcript, false).unwrap());
        assert_eq!(segment.text, "Hello");
        assert_eq!(segment.span, None);
        assert_eq!(segment.words, Vec::new());
    }

    #[test]
    fn words_outside_the_transcript_are_not_invented() {
        let mut transcript = so_my_friend();
        transcript.segments = vec![segment(2, 5, 0, 100, "friend")];
        let segment = only_segment(hypothesis(&transcript, false).unwrap());
        assert_eq!(segment.words, Vec::new());
        assert_eq!(segment.text, "friend");
    }

    #[test]
    fn silence_is_an_empty_hypothesis() {
        let silence = Transcript {
            timestamp_kind: TimestampKind::Token,
            ..Transcript::default()
        };
        assert_eq!(
            hypothesis(&silence, true).unwrap(),
            Hypothesis::empty(Segmentation::Synthesize)
        );
    }

    #[test]
    fn words_with_times_are_split_by_the_pipeline_and_coarser_timing_is_kept_as_given() {
        let kinds = [
            (TimestampKind::Token, Segmentation::Synthesize),
            (TimestampKind::Word, Segmentation::Synthesize),
            (TimestampKind::Segment, Segmentation::Native),
            (TimestampKind::None, Segmentation::Native),
        ];
        for (kind, expected) in kinds {
            let transcript = Transcript {
                timestamp_kind: kind,
                ..so_my_friend()
            };
            assert_eq!(
                hypothesis(&transcript, false).unwrap().segmentation,
                expected
            );
        }
    }

    #[test]
    fn text_without_times_is_one_untimed_segment() {
        let transcript = Transcript {
            text: "Bonjour tout le monde".to_owned(),
            timestamp_kind: TimestampKind::None,
            segments: vec![segment(0, 0, 0, 0, " Bonjour tout le monde")],
            ..Transcript::default()
        };
        let hypothesis = hypothesis(&transcript, false).unwrap();
        assert_eq!(hypothesis.segmentation, Segmentation::Native);
        let segment = only_segment(hypothesis);
        assert_eq!(segment.text, "Bonjour tout le monde");
        assert_eq!(segment.span, None);
        assert_eq!(segment.words, Vec::new());
    }

    #[test]
    fn the_language_is_reported_only_when_it_was_asked_for() {
        let mut transcript = so_my_friend();
        transcript.language = Some("en".to_owned());
        let detected = hypothesis(&transcript, true).unwrap().language;
        assert_eq!(
            detected,
            Some(LanguageEstimate {
                tag: LanguageTag::english(),
                confidence: None,
            })
        );
        assert_eq!(hypothesis(&transcript, false).unwrap().language, None);
    }

    #[test]
    fn a_library_without_a_language_gives_none() {
        assert_eq!(hypothesis(&so_my_friend(), true).unwrap().language, None);
    }

    #[test]
    fn a_language_that_is_no_tag_is_an_error() {
        let mut transcript = so_my_friend();
        transcript.language = Some("<|nospeech|>".to_owned());
        assert!(matches!(
            hypothesis(&transcript, true),
            Err(TranscribeEngineError::LanguageCode { .. })
        ));
    }
}
