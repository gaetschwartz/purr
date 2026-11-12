//! Where speech is in a streaming window, from whisper.cpp's Silero voice activity detector.

use crate::{error::TranscriptionError, whisper::commit::Span};
use std::path::Path;
use whisper_rs::{WhisperVadContext, WhisperVadContextParams, WhisperVadParams};

const SAMPLES_PER_CENTISECOND: f32 = 160.0;

/// whisper.cpp's Silero voice activity detector.
pub struct SpeechDetector {
    context: WhisperVadContext,
}

impl SpeechDetector {
    pub fn load(path: &Path) -> Result<Self, TranscriptionError> {
        let mut params = WhisperVadContextParams::new();
        params.set_n_threads(1);
        let context =
            WhisperVadContext::new(&path.to_string_lossy(), params).map_err(|source| {
                TranscriptionError::VadModel {
                    path: path.to_owned(),
                    source,
                }
            })?;
        Ok(Self { context })
    }

    /// The spans of `samples` that hold speech, in order.
    pub fn speech(&mut self, samples: &[f32]) -> Result<Vec<Span>, TranscriptionError> {
        let segments = self
            .context
            .segments_from_samples(WhisperVadParams::new(), samples)
            .map_err(|source| TranscriptionError::Failed { source })?;
        Ok(segments
            .map(|segment| Span {
                start: (segment.start.max(0.0) * SAMPLES_PER_CENTISECOND) as usize,
                end: ((segment.end.max(0.0) * SAMPLES_PER_CENTISECOND) as usize).min(samples.len()),
            })
            .collect())
    }
}

/// The latest point of a window that falls in a pause between words, where the window can end
/// without cutting one, at least `min_window` samples in.
///
/// Speech still going when the window ends counts as a pause only once `min_tail` samples of
/// silence follow it.
pub fn last_pause(
    speech: &[Span],
    len: usize,
    min_window: usize,
    min_tail: usize,
) -> Option<usize> {
    speech.iter().enumerate().rev().find_map(|(i, span)| {
        let next = speech.get(i + 1).map_or(len, |next| next.start);
        let gap = next.saturating_sub(span.end);
        let paused = if i + 1 < speech.len() {
            gap > 0
        } else {
            gap >= min_tail
        };
        let cut = span.end + gap / 2;
        (paused && cut >= min_window).then_some(cut)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: usize = 16_000;

    fn span(start: f32, end: f32) -> Span {
        Span {
            start: (start * S as f32) as usize,
            end: (end * S as f32) as usize,
        }
    }

    #[test]
    fn cuts_in_the_middle_of_the_latest_pause() {
        let speech = [span(0.5, 3.0), span(4.0, 7.0), span(7.4, 9.9)];
        assert_eq!(last_pause(&speech, 10 * S, S, S), Some(span(0.0, 7.2).end));
    }

    #[test]
    fn speech_that_ended_long_enough_before_the_window_end_is_a_pause() {
        let speech = [span(0.5, 3.0), span(4.0, 7.0)];
        assert_eq!(last_pause(&speech, 10 * S, S, S), Some(span(0.0, 8.5).end));
    }

    #[test]
    fn never_cuts_before_the_least_window() {
        let speech = [span(0.1, 0.6), span(0.8, 9.9)];
        assert_eq!(last_pause(&speech, 10 * S, S, S), None);
    }

    #[test]
    fn continuous_speech_has_no_pause() {
        assert_eq!(last_pause(&[span(0.0, 9.95)], 10 * S, S, S), None);
        assert_eq!(last_pause(&[], 10 * S, S, S), None);
    }
}
