//! A Parakeet model behind the pipeline's window decoder.

use std::time::Duration;

use purr_common::{
    language::{support::LanguageSupport, tag::LanguageTag},
    model::profile::InputLimit,
};
use purr_core::engine::{
    decoder::{DecodeRequest, LanguageRequest, WindowDecoder, WindowLimits},
    error::DecodeError,
    hypothesis::{Completion, Decoded, Hypothesis, Segmentation},
};
use transcribe_cpp::{Capabilities, Model, RunOptions, Session, SessionOptions, TimestampKind};

use crate::{
    error::{TranscribeEngineError, decode_error},
    transcript::hypothesis,
};

/// Parakeet is trained on stretches of speech up to this long; past it, a decode costs more than
/// the context it gains
const STREAMING_WINDOW: Duration = Duration::from_secs(30);

/// A Parakeet model loaded on one device.
pub struct ParakeetDecoder {
    session: Session,
    languages: LanguageSupport,
    input: InputLimit,
}

impl ParakeetDecoder {
    pub fn new(model: &Model, threads: usize) -> Result<Self, TranscribeEngineError> {
        let capabilities = model.capabilities()?;
        let session = model.session_with(&SessionOptions {
            n_threads: i32::try_from(threads).unwrap_or(i32::MAX),
            ..SessionOptions::default()
        })?;
        Ok(Self {
            session,
            languages: languages(&capabilities)?,
            input: input_limit(&capabilities),
        })
    }
}

/// The languages the model lists; a model that lists none is not tied to any.
fn languages(capabilities: &Capabilities) -> Result<LanguageSupport, TranscribeEngineError> {
    if capabilities.languages.is_empty() {
        return Ok(LanguageSupport::Any);
    }
    let tags = capabilities
        .languages
        .iter()
        .map(|code| {
            LanguageTag::parse(code).map_err(|source| TranscribeEngineError::LanguageCode {
                code: code.clone(),
                source,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LanguageSupport::Only(tags.into()))
}

fn input_limit(capabilities: &Capabilities) -> InputLimit {
    u64::try_from(capabilities.max_audio_ms)
        .ok()
        .filter(|milliseconds| *milliseconds > 0)
        .map_or(InputLimit::Unbounded, |milliseconds| {
            InputLimit::UpTo(Duration::from_millis(milliseconds))
        })
}

impl WindowDecoder for ParakeetDecoder {
    fn limits(&self) -> WindowLimits {
        WindowLimits {
            input: self.input,
            window: STREAMING_WINDOW,
        }
    }

    fn supports(&self, language: &LanguageTag) -> bool {
        self.languages.supports(language)
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
        if let LanguageRequest::Use(tag) = request.language
            && !self.supports(tag)
        {
            return Err(DecodeError::UnsupportedLanguage(tag.clone()));
        }
        let options = RunOptions {
            timestamps: TimestampKind::Token,
            ..RunOptions::default()
        };
        let transcript = self
            .session
            .run(request.pcm, &options)
            .map_err(decode_error)?;
        let detect = matches!(request.language, LanguageRequest::Detect);
        Ok(Decoded {
            hypothesis: hypothesis(&transcript, detect)
                .map_err(|error| DecodeError::Engine(Box::new(error)))?,
            completion: Completion::Complete,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capabilities(languages: &[&str], max_audio_ms: i64) -> Capabilities {
        Capabilities {
            native_sample_rate: 16_000,
            languages: languages.iter().map(|code| (*code).to_owned()).collect(),
            translate_target_languages: Vec::new(),
            max_timestamp_kind: TimestampKind::Token,
            supports_language_detect: true,
            supports_translate: false,
            supports_streaming: false,
            supports_spec_decode: false,
            max_audio_ms,
        }
    }

    fn tag(text: &str) -> LanguageTag {
        text.parse().unwrap()
    }

    #[test]
    fn the_model_decides_which_languages_it_is_asked_in() {
        let support = languages(&capabilities(&["de", "fr"], 0)).unwrap();
        assert!(support.supports(&tag("de")));
        assert!(support.supports(&tag("fr-CH")));
        assert!(!support.supports(&tag("ja")));
    }

    #[test]
    fn a_model_that_lists_no_language_takes_any() {
        assert_eq!(
            languages(&capabilities(&[], 0)).unwrap(),
            LanguageSupport::Any
        );
    }

    #[test]
    fn a_language_code_that_is_no_tag_is_refused() {
        assert!(matches!(
            languages(&capabilities(&["de", "<|x|>"], 0)),
            Err(TranscribeEngineError::LanguageCode { .. })
        ));
    }

    #[test]
    fn no_maximum_means_audio_of_any_length() {
        assert_eq!(input_limit(&capabilities(&[], 0)), InputLimit::Unbounded);
        assert_eq!(
            input_limit(&capabilities(&[], 90_000)),
            InputLimit::UpTo(Duration::from_secs(90))
        );
        assert_eq!(input_limit(&capabilities(&[], -1)), InputLimit::Unbounded);
    }
}
