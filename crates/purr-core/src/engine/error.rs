use purr_common::language::tag::LanguageTag;

use crate::accel::report::AttemptFailure;

/// An error of the engine's own, in its own words.
pub type EngineError = Box<dyn std::error::Error + Send + Sync>;

/// Why an engine could not load a model on a device.
#[oopsie::oopsie]
pub enum OpenError {
    /// The engine does not list the device it was asked to use
    #[oopsie("the engine lists no such device")]
    NoDevice,
    #[oopsie(transparent)]
    Device { source: AttemptFailure },
    #[oopsie(transparent)]
    Engine { source: EngineError },
}

/// Why an engine could not decode a window.
#[oopsie::oopsie]
pub enum DecodeError {
    #[oopsie("the audio is longer than the model takes at once")]
    InputTooLong,
    #[oopsie("the audio is too short to decode")]
    InputTooShort,
    #[oopsie("the model does not know the language `{language}`")]
    UnsupportedLanguage { language: LanguageTag },
    #[oopsie("the device ran out of memory")]
    OutOfMemory,
    #[oopsie(transparent)]
    Engine { source: EngineError },
}

/// Why a speech detector could not look at a window.
#[oopsie::oopsie]
#[oopsie(transparent)]
pub struct DetectError {
    pub source: EngineError,
}
