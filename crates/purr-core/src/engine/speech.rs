use super::{error::DetectError, hypothesis::Span};

/// Finds where speech is in a window of audio.
pub trait SpeechDetector: Send {
    /// The spans of `pcm` (mono, 16 kHz) that hold speech, in order.
    fn speech(&mut self, pcm: &[f32]) -> Result<Vec<Span>, DetectError>;
}
