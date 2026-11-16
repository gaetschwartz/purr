use purr_common::language::tag::LanguageTagError;
use purr_core::{
    accel::{plan::Candidate, report::AttemptFailure},
    engine::{
        error::{DecodeError, DetectError, OpenError},
        hypothesis::Completion,
    },
};
use transcribe_cpp::Transcript;

use crate::runtime::Unavailable;

/// What goes wrong inside the transcribe.cpp engine.
#[oopsie::oopsie]
pub enum TranscribeEngineError {
    #[oopsie("transcribe.cpp cannot be used")]
    Library { source: Unavailable },

    #[oopsie(transparent)]
    Transcribe { source: transcribe_cpp::Error },

    #[oopsie("transcribe.cpp reports the language `{code}`, which is not a language tag")]
    LanguageCode {
        code: String,
        source: LanguageTagError,
    },
}

impl From<TranscribeEngineError> for OpenError {
    fn from(error: TranscribeEngineError) -> Self {
        Self::Engine {
            source: Box::new(error),
        }
    }
}

impl From<TranscribeEngineError> for DetectError {
    fn from(error: TranscribeEngineError) -> Self {
        Self {
            source: Box::new(error),
        }
    }
}

/// Why the model could not be loaded on the device `candidate` stands for.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "the library's error enum is non-exhaustive"
)]
pub fn open_error(error: TranscribeEngineError, candidate: &Candidate) -> OpenError {
    let device = candidate.name.clone();
    match error {
        TranscribeEngineError::Transcribe {
            source: transcribe_cpp::Error::Backend(_),
            ..
        } => AttemptFailure::DeviceInit { device }.into(),
        TranscribeEngineError::Transcribe {
            source: transcribe_cpp::Error::OutOfMemory(_),
            ..
        } if candidate.backend.is_gpu() => AttemptFailure::DeviceOutOfMemory { device }.into(),
        error => error.into(),
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
        error => DecodeError::Engine {
            source: Box::new(TranscribeEngineError::from(error)),
        },
    }
}

/// What a run gave: its transcript and whether the library ended it before the end of the audio,
/// which leaves the text so far as the transcript.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "the library's error enum is non-exhaustive"
)]
pub fn finished(
    run: Result<Transcript, transcribe_cpp::Error>,
) -> Result<(Transcript, Completion), DecodeError> {
    match run {
        Ok(transcript) => Ok((transcript, Completion::Complete)),
        Err(transcribe_cpp::Error::OutputTruncated { partial, .. }) => Ok((
            partial.map(|transcript| *transcript).unwrap_or_default(),
            Completion::Truncated,
        )),
        Err(transcribe_cpp::Error::OutputRepetition { partial, .. }) => Ok((
            partial.map(|transcript| *transcript).unwrap_or_default(),
            Completion::Repeating,
        )),
        Err(error) => Err(decode_error(error)),
    }
}

#[cfg(test)]
mod tests {
    use purr_common::devices::DeviceKey;
    use purr_core::accel::backend::AccelBackend;

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

    fn partial(text: &str) -> Option<Box<Transcript>> {
        Some(Box::new(Transcript {
            text: text.to_owned(),
            ..Transcript::default()
        }))
    }

    #[test]
    fn a_run_cut_short_gives_the_text_before_the_cut_and_says_why() {
        let truncated = finished(Err(transcribe_cpp::Error::OutputTruncated {
            message: message(),
            partial: partial("so far"),
        }))
        .unwrap();
        assert_eq!(truncated.0.text, "so far");
        assert_eq!(truncated.1, Completion::Truncated);
        let repeating = finished(Err(transcribe_cpp::Error::OutputRepetition {
            message: message(),
            partial: partial("loop"),
        }))
        .unwrap();
        assert_eq!(repeating.0.text, "loop");
        assert_eq!(repeating.1, Completion::Repeating);
    }

    #[test]
    fn a_run_cut_short_without_a_partial_transcript_gave_nothing() {
        let (transcript, completion) = finished(Err(transcribe_cpp::Error::OutputTruncated {
            message: message(),
            partial: None,
        }))
        .unwrap();
        assert_eq!(transcript, Transcript::default());
        assert_eq!(completion, Completion::Truncated);
    }

    #[test]
    fn a_finished_run_is_complete() {
        let (transcript, completion) = finished(Ok(Transcript {
            text: "all of it".to_owned(),
            ..Transcript::default()
        }))
        .unwrap();
        assert_eq!(transcript.text, "all of it");
        assert_eq!(completion, Completion::Complete);
    }

    #[test]
    fn any_other_failure_of_a_run_is_an_error() {
        assert!(matches!(
            finished(Err(transcribe_cpp::Error::Busy(message()))),
            Err(DecodeError::Engine { .. })
        ));
        assert!(matches!(
            finished(Err(transcribe_cpp::Error::InputTooLong(message()))),
            Err(DecodeError::InputTooLong)
        ));
    }

    fn candidate(backend: AccelBackend, name: &str) -> Candidate {
        Candidate {
            key: DeviceKey::cpu(),
            backend,
            name: name.to_owned(),
            pci_id: None,
        }
    }

    fn open(error: transcribe_cpp::Error, candidate: &Candidate) -> OpenError {
        open_error(error.into(), candidate)
    }

    #[test]
    fn a_backend_that_fails_to_load_the_model_lets_the_search_go_on() {
        assert!(matches!(
            open(
                transcribe_cpp::Error::Backend(message()),
                &candidate(AccelBackend::Vulkan, "Card")
            ),
            OpenError::Device { source: AttemptFailure::DeviceInit { device } } if device == "Card"
        ));
    }

    #[test]
    fn a_gpu_without_room_for_the_model_lets_the_search_go_on() {
        assert!(matches!(
            open(
                transcribe_cpp::Error::OutOfMemory(message()),
                &candidate(AccelBackend::Cuda, "Card")
            ),
            OpenError::Device { source: AttemptFailure::DeviceOutOfMemory { device } } if device == "Card"
        ));
    }

    #[test]
    fn a_processor_without_room_for_the_model_stops_the_search() {
        assert!(matches!(
            open(
                transcribe_cpp::Error::OutOfMemory(message()),
                &candidate(AccelBackend::Cpu, "CPU")
            ),
            OpenError::Engine { .. }
        ));
    }

    #[test]
    fn a_model_that_cannot_be_read_stops_the_search() {
        for error in [
            transcribe_cpp::Error::ModelLoad(message()),
            transcribe_cpp::Error::ModelFileNotFound(message()),
        ] {
            assert!(matches!(
                open(error, &candidate(AccelBackend::Vulkan, "Card")),
                OpenError::Engine { .. }
            ));
        }
    }
}
