//! The JSON the CLI prints: one object per streamed chunk, or one document without streaming.

use purr_common::word::Word;
use purr_core::{
    StreamingChunk, SyncTranscriptionResult, accel::report::AccelReport,
    whisper::TranscriptionStats,
};
use serde::Serialize;

#[derive(Debug, Serialize, PartialEq)]
pub struct JsonWord<'a> {
    pub text: &'a str,
    /// Seconds
    pub start: Option<f64>,
    pub end: Option<f64>,
    /// Probability of the word's least certain token, rounded to three decimals
    pub score: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct JsonChunk<'a> {
    text: &'a str,
    start: f64,
    end: f64,
    is_final: bool,
    chunk_index: usize,
    final_stats: Option<&'a TranscriptionStats>,
    language: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    words: Option<Vec<JsonWord<'a>>>,
}

#[derive(Debug, Serialize)]
struct JsonSegment<'a> {
    text: &'a str,
    start: f64,
    end: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    words: Option<Vec<JsonWord<'a>>>,
}

#[derive(Debug, Serialize)]
pub struct JsonDocument<'a> {
    text: &'a str,
    segments: Vec<JsonSegment<'a>>,
    processing_time: f64,
    audio_duration: f32,
    stats: &'a TranscriptionStats,
    accel: &'a AccelReport,
}

/// The words of `text` as JSON objects; a word whose range falls outside `text` is left out.
#[must_use]
pub fn words<'a>(text: &'a str, words: &[Word]) -> Vec<JsonWord<'a>> {
    words
        .iter()
        .filter_map(|word| {
            Some(JsonWord {
                text: word.text(text)?,
                start: word.start_seconds(),
                end: word.end_seconds(),
                score: word.probability().map(|p| (p * 1000.0).round() / 1000.0),
            })
        })
        .collect()
}

/// One streamed chunk, with its words when `with_words`.
#[must_use]
pub fn chunk(chunk: &StreamingChunk, with_words: bool) -> JsonChunk<'_> {
    JsonChunk {
        text: &chunk.text,
        start: chunk.start,
        end: chunk.end,
        is_final: chunk.is_final,
        chunk_index: chunk.chunk_index,
        final_stats: chunk.final_stats.as_ref(),
        language: chunk.language.as_deref(),
        words: with_words.then(|| words(&chunk.text, &chunk.words)),
    }
}

/// A whole transcription, with the words of each segment when `with_words`.
#[must_use]
pub fn document(result: &SyncTranscriptionResult, with_words: bool) -> JsonDocument<'_> {
    JsonDocument {
        text: &result.text,
        segments: result
            .segments
            .iter()
            .map(|segment| JsonSegment {
                text: &segment.text,
                start: segment.start,
                end: segment.end,
                words: with_words.then(|| words(&segment.text, &segment.words)),
            })
            .collect(),
        processing_time: result.processing_time,
        audio_duration: result.audio_duration,
        stats: &result.stats,
        accel: &result.accel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_read_their_text_times_and_rounded_score() {
        let found = words(
            "ask not",
            &[
                Word {
                    range: 0..3,
                    start_cs: Some(349),
                    end_cs: Some(424),
                    score: Some(191),
                },
                Word {
                    range: 4..7,
                    start_cs: None,
                    end_cs: None,
                    score: None,
                },
                Word {
                    range: 4..12,
                    start_cs: None,
                    end_cs: None,
                    score: None,
                },
            ],
        );
        assert_eq!(
            found,
            vec![
                JsonWord {
                    text: "ask",
                    start: Some(3.49),
                    end: Some(4.24),
                    score: Some(0.749),
                },
                JsonWord {
                    text: "not",
                    start: None,
                    end: None,
                    score: None,
                },
            ]
        );
    }
}
