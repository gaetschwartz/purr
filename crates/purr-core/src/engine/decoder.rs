use std::time::Duration;

use purr_common::{
    language::tag::LanguageTag,
    model::profile::{InputLimit, Timing},
};

use super::{cancel::CancelToken, error::DecodeError, hypothesis::Decoded};

/// The language a decode runs in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LanguageRequest {
    /// Find it in the window and report it with the hypothesis
    Detect,
    Use(LanguageTag),
}

/// One window of audio to recognise.
#[derive(Debug, Clone, Copy)]
pub struct DecodeRequest<'a> {
    /// Mono samples at 16 kHz
    pub pcm: &'a [f32],
    pub language: &'a LanguageRequest,
    /// The finest timing wanted; finer timing may cost more
    pub timing: Timing,
    pub cancel: &'a CancelToken,
}

/// How much audio a decoder takes at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowLimits {
    /// The longest audio one decode accepts
    pub input: InputLimit,
    /// The longest window worth decoding while streaming: past it, a decode costs more than it
    /// gains in context
    pub window: Duration,
}

/// A model loaded on a device, ready to recognise windows of audio one after the other.
pub trait WindowDecoder: Send {
    fn limits(&self) -> WindowLimits;

    /// Whether the model can be asked to decode in `language`.
    fn supports(&self, language: &LanguageTag) -> bool;

    fn decode(&mut self, request: DecodeRequest<'_>) -> Result<Decoded, DecodeError>;
}
