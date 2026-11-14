//! The segments of the last `full()` call, with their words.

use crate::engine::hypothesis::{Segment, Span};
use tracing::debug;
use whisper_rs::WhisperState;

use super::{
    error::WhisperEngineError,
    words::{Token, group},
};

const SAMPLES_PER_CENTISECOND: usize = 160;

/// Reads every segment `state` holds; `eot` is the model's end-of-text token id.
pub fn read(state: &WhisperState, eot: i32) -> Result<Vec<Segment>, WhisperEngineError> {
    let mut decoded = Vec::new();
    for segment in state.as_iter() {
        let raw = segment.to_str_lossy()?;
        let (start_cs, end_cs) = (segment.start_timestamp(), segment.end_timestamp());
        let mut tokens = Vec::new();
        for index in 0..segment.n_tokens() {
            let Some(token) = segment.get_token(index) else {
                continue;
            };
            let data = token.token_data();
            tokens.push(Token {
                id: data.id,
                bytes: token.to_bytes()?,
                p: data.p,
                t0: data.t0,
                t1: data.t1,
            });
        }
        let (text, words) = if let Some(grouped) = group(&raw, &tokens, eot, (start_cs, end_cs)) {
            (grouped.text, grouped.words)
        } else {
            debug!(
                "The tokens of segment {} do not spell its text; it has no words",
                segment.segment_index()
            );
            (raw.trim().to_owned(), Vec::new())
        };
        decoded.push(Segment {
            text,
            span: Some(Span {
                start: start_cs.max(0) as usize * SAMPLES_PER_CENTISECOND,
                end: end_cs.max(0) as usize * SAMPLES_PER_CENTISECOND,
            }),
            words,
        });
    }
    Ok(decoded)
}
