//! The segments of the last `full()` call, with their words.

use crate::{
    error::{Result, TranscriptionError},
    whisper::words::{Token, group},
};
use purr_common::word::Word;
use tracing::debug;
use whisper_rs::WhisperState;

/// A segment of the last decoded window, timed in centiseconds from the window start.
pub struct DecodedSegment {
    pub start_cs: i64,
    pub end_cs: i64,
    /// The trimmed text
    pub text: String,
    /// The words of `text`, timed from the window start
    pub words: Vec<Word>,
}

/// Reads every segment `state` holds; `eot` is the model's end-of-text token id.
pub fn read(state: &WhisperState, eot: i32) -> Result<Vec<DecodedSegment>> {
    let mut decoded = Vec::new();
    for segment in state.as_iter() {
        let raw = segment
            .to_str_lossy()
            .map_err(|source| TranscriptionError::Failed { source })?;
        let (start_cs, end_cs) = (segment.start_timestamp(), segment.end_timestamp());
        let mut tokens = Vec::new();
        for index in 0..segment.n_tokens() {
            let Some(token) = segment.get_token(index) else {
                continue;
            };
            let data = token.token_data();
            tokens.push(Token {
                id: data.id,
                bytes: token
                    .to_bytes()
                    .map_err(|source| TranscriptionError::Failed { source })?,
                p: data.p,
                t0: data.t0,
                t1: data.t1,
            });
        }
        let (text, words) = match group(&raw, &tokens, eot, (start_cs, end_cs)) {
            Some(grouped) => (grouped.text, grouped.words),
            None => {
                debug!(
                    "The tokens of segment {} do not spell its text; it has no words",
                    segment.segment_index()
                );
                (raw.trim().to_owned(), Vec::new())
            }
        };
        decoded.push(DecodedSegment {
            start_cs,
            end_cs,
            text,
            words,
        });
    }
    Ok(decoded)
}
