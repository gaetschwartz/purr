//! A transcribe.cpp model behind the pipeline's window decoder.

use std::time::Duration;

use purr_common::{language::tag::LanguageTag, model::profile::InputLimit};
use purr_core::engine::{
    decoder::{DecodeRequest, LanguageRequest, WindowDecoder, WindowLimits},
    error::DecodeError,
    hypothesis::{Completion, Decoded, Hypothesis, Segmentation},
};
use transcribe_cpp::{Model, RunOptions, Session, SessionOptions, TimestampKind};

use crate::{
    error::{TranscribeEngineError, finished},
    languages::ModelLanguages,
    transcript::hypothesis,
};

/// The longest window worth decoding while streaming: past it, a decode costs more than it
/// gains in context
const STREAMING_WINDOW: Duration = Duration::from_secs(30);

/// A transcribe.cpp model loaded on one device.
///
/// What it asks of the model and what it reports follows what the model says it can do: the
/// languages it lists, the longest audio it takes and the finest timestamps it produces.
pub struct TranscribeDecoder {
    session: Session,
    languages: ModelLanguages,
    input: InputLimit,
}

impl TranscribeDecoder {
    pub fn new(model: &Model, threads: usize) -> Result<Self, TranscribeEngineError> {
        let capabilities = model.capabilities()?;
        let session = model.session_with(&SessionOptions {
            n_threads: i32::try_from(threads).unwrap_or(i32::MAX),
            ..SessionOptions::default()
        })?;
        let limits = session.limits()?;
        Ok(Self {
            session,
            languages: ModelLanguages::from_codes(&capabilities.languages)?,
            input: input_limit(limits.effective_max_audio_ms),
        })
    }
}

fn input_limit(max_audio_ms: i64) -> InputLimit {
    u64::try_from(max_audio_ms)
        .ok()
        .filter(|milliseconds| *milliseconds > 0)
        .map_or(InputLimit::Unbounded, |milliseconds| {
            InputLimit::UpTo(Duration::from_millis(milliseconds))
        })
}

impl WindowDecoder for TranscribeDecoder {
    fn limits(&self) -> WindowLimits {
        WindowLimits {
            input: self.input,
            window: STREAMING_WINDOW,
        }
    }

    fn supports(&self, language: &LanguageTag) -> bool {
        self.languages.support().supports(language)
    }

    fn decode(&mut self, request: DecodeRequest<'_>) -> Result<Decoded, DecodeError> {
        if request.cancel.is_cancelled() {
            return Ok(Decoded {
                hypothesis: Hypothesis::empty(Segmentation::Synthesize),
                completion: Completion::Cancelled,
            });
        }
        if request.pcm.is_empty() {
            return Err(DecodeError::InputTooShort);
        }
        let language = match request.language {
            LanguageRequest::Detect => None,
            LanguageRequest::Use(tag) if self.supports(tag) => {
                self.languages.code(tag).map(str::to_owned)
            }
            LanguageRequest::Use(tag) => {
                return Err(DecodeError::UnsupportedLanguage {
                    language: tag.clone(),
                });
            }
        };
        let options = RunOptions {
            timestamps: TimestampKind::Auto,
            language,
            ..RunOptions::default()
        };
        let (transcript, completion) = finished(self.session.run(request.pcm, &options))?;
        let detect = matches!(request.language, LanguageRequest::Detect);
        Ok(Decoded {
            hypothesis: hypothesis(&transcript, detect).map_err(|error| DecodeError::Engine {
                source: Box::new(error),
            })?,
            completion,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_maximum_means_audio_of_any_length() {
        assert_eq!(input_limit(0), InputLimit::Unbounded);
        assert_eq!(
            input_limit(90_000),
            InputLimit::UpTo(Duration::from_secs(90))
        );
        assert_eq!(input_limit(-1), InputLimit::Unbounded);
    }
}
