//! Where speech is in a streaming window, from transcribe.cpp's Silero voice activity detector.

use std::path::Path;

use purr_core::engine::{error::DetectError, hypothesis::Span, speech::SpeechDetector};
use transcribe_cpp::{
    Backend, Model, ModelOptions, VadOptions, VadSegment, VadSession, VadSessionOptions,
};

use crate::error::TranscribeEngineError;

/// transcribe.cpp's Silero voice activity detector.
pub struct SileroDetector {
    session: VadSession,
}

impl SileroDetector {
    pub fn load(path: &Path) -> Result<Self, TranscribeEngineError> {
        let model = Model::load_with(
            path,
            &ModelOptions {
                backend: Backend::Cpu,
                ..ModelOptions::default()
            },
        )?;
        let session = model.vad_session_with(&VadSessionOptions { n_threads: 1 })?;
        Ok(Self { session })
    }
}

impl SpeechDetector for SileroDetector {
    fn speech(&mut self, pcm: &[f32]) -> Result<Vec<Span>, DetectError> {
        let result = self
            .session
            .run(pcm, &VadOptions::default())
            .map_err(TranscribeEngineError::from)?;
        Ok(spans(&result.segments, pcm.len()))
    }
}

/// The segments as spans of a window of `len` samples.
fn spans(segments: &[VadSegment], len: usize) -> Vec<Span> {
    let clamp = |sample: i64| usize::try_from(sample.max(0)).map_or(len, |sample| sample.min(len));
    segments
        .iter()
        .map(|segment| Span {
            start: clamp(segment.start_sample),
            end: clamp(segment.end_sample),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(start_sample: i64, end_sample: i64) -> VadSegment {
        let mut segment = VadSegment::default();
        segment.start_sample = start_sample;
        segment.end_sample = end_sample;
        segment
    }

    #[test]
    fn speech_stays_inside_the_window() {
        assert_eq!(
            spans(&[segment(-30, 8_000), segment(16_000, 40_000)], 32_000),
            [
                Span {
                    start: 0,
                    end: 8_000
                },
                Span {
                    start: 16_000,
                    end: 32_000
                },
            ]
        );
    }

    #[test]
    fn no_segments_is_no_speech() {
        assert_eq!(spans(&[], 16_000), Vec::new());
    }
}
