//! Where speech is in a streaming window, from whisper.cpp's Silero voice activity detector.

use std::path::Path;

use whisper_rs::{WhisperVadContext, WhisperVadContextParams, WhisperVadParams};

use purr_core::engine::{error::DetectError, hypothesis::Span, speech::SpeechDetector};

use crate::error::WhisperEngineError;

const SAMPLES_PER_CENTISECOND: f32 = 160.0;

/// whisper.cpp's Silero voice activity detector.
pub struct SileroDetector {
    context: WhisperVadContext,
}

impl SileroDetector {
    pub fn load(path: &Path) -> Result<Self, WhisperEngineError> {
        let mut params = WhisperVadContextParams::new();
        params.set_n_threads(1);
        let context = WhisperVadContext::new(&path.to_string_lossy(), params)?;
        Ok(Self { context })
    }
}

impl SpeechDetector for SileroDetector {
    fn speech(&mut self, pcm: &[f32]) -> Result<Vec<Span>, DetectError> {
        let segments = self
            .context
            .segments_from_samples(WhisperVadParams::new(), pcm)
            .map_err(WhisperEngineError::from)?;
        Ok(segments
            .map(|segment| Span {
                start: (segment.start.max(0.0) * SAMPLES_PER_CENTISECOND) as usize,
                end: ((segment.end.max(0.0) * SAMPLES_PER_CENTISECOND) as usize).min(pcm.len()),
            })
            .collect())
    }
}
