use purr_common::language::tag::LanguageTag;

use crate::accel::report::AttemptFailure;

/// An error of the engine's own, in its own words.
pub type EngineError = Box<dyn std::error::Error + Send + Sync>;

/// Why an engine could not load a model on a device.
#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    /// The engine does not list the device it was asked to use
    #[error("the engine lists no such device")]
    NoDevice,
    #[error(transparent)]
    Device(#[from] AttemptFailure),
    #[error(transparent)]
    Engine(EngineError),
}

/// Why an engine could not decode a window.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("the audio is longer than the model takes at once")]
    InputTooLong,
    #[error("the audio is too short to decode")]
    InputTooShort,
    #[error("the model does not know the language `{0}`")]
    UnsupportedLanguage(LanguageTag),
    #[error("the device ran out of memory")]
    OutOfMemory,
    #[error(transparent)]
    Engine(EngineError),
}

/// Why a speech detector could not look at a window.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct DetectError(pub EngineError);
