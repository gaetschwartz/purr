use purr_common::language::tag::LanguageTagError;
use purr_core::{
    accel::report::AttemptFailure,
    engine::error::{DecodeError, DetectError, OpenError},
};

use crate::runtime::Unavailable;

/// What goes wrong inside the transcribe.cpp engine.
#[derive(Debug, thiserror::Error)]
pub enum TranscribeEngineError {
    #[error("transcribe.cpp cannot be used")]
    Library(#[source] &'static Unavailable),

    #[error(transparent)]
    Transcribe(#[from] transcribe_cpp::Error),

    #[error("transcribe.cpp reports the language `{code}`, which is not a language tag")]
    LanguageCode {
        code: String,
        #[source]
        source: LanguageTagError,
    },
}

impl From<TranscribeEngineError> for OpenError {
    fn from(error: TranscribeEngineError) -> Self {
        Self::Engine(Box::new(error))
    }
}

impl From<TranscribeEngineError> for DetectError {
    fn from(error: TranscribeEngineError) -> Self {
        Self(Box::new(error))
    }
}

/// Why the model could not be loaded on the device called `device`.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "the library's error enum is non-exhaustive"
)]
pub fn open_error(error: transcribe_cpp::Error, device: &str) -> OpenError {
    match error {
        transcribe_cpp::Error::Backend(_) => AttemptFailure::DeviceInit {
            device: device.to_owned(),
        }
        .into(),
        error => TranscribeEngineError::from(error).into(),
    }
}

/// What a failed run means for the window being decoded.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "the library's error enum is non-exhaustive"
)]
pub fn decode_error(error: transcribe_cpp::Error) -> DecodeError {
    match error {
        transcribe_cpp::Error::InputTooLong(_) => DecodeError::InputTooLong,
        transcribe_cpp::Error::InputTooShort(_) => DecodeError::InputTooShort,
        transcribe_cpp::Error::OutOfMemory(_) => DecodeError::OutOfMemory,
        error => DecodeError::Engine(Box::new(TranscribeEngineError::from(error))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message() -> String {
        "run".to_owned()
    }

    #[test]
    fn the_limits_of_the_model_are_typed_decode_errors() {
        assert!(matches!(
            decode_error(transcribe_cpp::Error::InputTooLong(message())),
            DecodeError::InputTooLong
        ));
        assert!(matches!(
            decode_error(transcribe_cpp::Error::InputTooShort(message())),
            DecodeError::InputTooShort
        ));
        assert!(matches!(
            decode_error(transcribe_cpp::Error::OutOfMemory(message())),
            DecodeError::OutOfMemory
        ));
    }

    #[test]
    fn an_incomplete_transcript_is_an_error_not_a_silent_result() {
        for error in [
            transcribe_cpp::Error::OutputTruncated {
                message: message(),
                partial: None,
            },
            transcribe_cpp::Error::OutputRepetition {
                message: message(),
                partial: None,
            },
            transcribe_cpp::Error::Busy(message()),
        ] {
            assert!(matches!(decode_error(error), DecodeError::Engine(_)));
        }
    }

    #[test]
    fn a_backend_that_fails_to_load_the_model_lets_the_search_go_on() {
        assert!(matches!(
            open_error(transcribe_cpp::Error::Backend(message()), "CPU"),
            OpenError::Device(AttemptFailure::DeviceInit { device }) if device == "CPU"
        ));
    }

    #[test]
    fn a_model_that_cannot_be_read_stops_the_search() {
        for error in [
            transcribe_cpp::Error::ModelLoad(message()),
            transcribe_cpp::Error::ModelFileNotFound(message()),
            transcribe_cpp::Error::OutOfMemory(message()),
        ] {
            assert!(matches!(open_error(error, "CPU"), OpenError::Engine(_)));
        }
    }
}
