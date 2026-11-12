//! How far a transcription has come, apart from the text it produced.

use serde::{Deserialize, Serialize};

/// What a transcription is doing now.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressPhase {
    Downloading,
    LoadingModel,
    DetectingLanguage,
    Transcribing,
}

/// One progress update of a running transcription.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct ProgressFrame {
    /// Seconds of the source, from its start, that the engine is done with and will not decode
    /// again; it never decreases and never passes the audio the engine has received
    pub transcribed_to: f64,
    /// The seconds of the source being decoded now, when the engine says
    pub decoding: Option<(f64, f64)>,
    pub phase: ProgressPhase,
}
