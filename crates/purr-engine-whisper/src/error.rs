use purr_core::engine::error::{DecodeError, DetectError, OpenError};

/// What goes wrong inside whisper.cpp.
#[oopsie::oopsie]
pub enum WhisperEngineError {
    #[oopsie(transparent)]
    Whisper { source: whisper_rs::WhisperError },

    #[oopsie("Whisper detected a language id it has no code for: {id}")]
    UnknownLanguageId { id: i32 },
}

impl From<WhisperEngineError> for DecodeError {
    fn from(error: WhisperEngineError) -> Self {
        match error {
            WhisperEngineError::Whisper {
                source: whisper_rs::WhisperError::NoSamples,
            } => Self::InputTooShort,
            WhisperEngineError::Whisper { .. } | WhisperEngineError::UnknownLanguageId { .. } => {
                Self::Engine {
                    source: Box::new(error),
                }
            }
        }
    }
}

impl From<WhisperEngineError> for OpenError {
    fn from(error: WhisperEngineError) -> Self {
        Self::Engine {
            source: Box::new(error),
        }
    }
}

impl From<WhisperEngineError> for DetectError {
    fn from(error: WhisperEngineError) -> Self {
        Self {
            source: Box::new(error),
        }
    }
}
