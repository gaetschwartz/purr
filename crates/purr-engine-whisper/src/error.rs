use purr_core::engine::error::{DecodeError, DetectError, OpenError};

/// What goes wrong inside whisper.cpp.
#[derive(Debug, thiserror::Error)]
pub enum WhisperEngineError {
    #[error(transparent)]
    Whisper(#[from] whisper_rs::WhisperError),

    #[error("Whisper detected a language id it has no code for: {id}")]
    UnknownLanguageId { id: i32 },
}

impl From<WhisperEngineError> for DecodeError {
    fn from(error: WhisperEngineError) -> Self {
        match error {
            WhisperEngineError::Whisper(whisper_rs::WhisperError::NoSamples) => Self::InputTooShort,
            WhisperEngineError::Whisper(_) | WhisperEngineError::UnknownLanguageId { .. } => {
                Self::Engine(Box::new(error))
            }
        }
    }
}

impl From<WhisperEngineError> for OpenError {
    fn from(error: WhisperEngineError) -> Self {
        Self::Engine(Box::new(error))
    }
}

impl From<WhisperEngineError> for DetectError {
    fn from(error: WhisperEngineError) -> Self {
        Self(Box::new(error))
    }
}
