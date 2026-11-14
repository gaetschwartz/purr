//! Whisper behind the pipeline's window decoder.

use std::{ffi::c_void, time::Duration};

use purr_common::{language::tag::LanguageTag, model::profile::InputLimit};
use whisper_rs::{FullParams, SamplingStrategy, WhisperState};

use crate::engine::{
    cancel::CancelToken,
    decoder::{DecodeRequest, LanguageRequest, WindowDecoder, WindowLimits},
    error::DecodeError,
    hypothesis::{Completion, Decoded, Hypothesis, LanguageEstimate, Segmentation},
};

use super::{
    error::WhisperEngineError,
    language::{detected_tag, whisper_code},
    segments,
};

/// Beams of the beam search, as whisper.cpp's own command line uses
const BEAM_SIZE: i32 = 5;
/// Whisper's encoder always sees 30 s, so a shorter window costs about as much to decode
const ENCODER_WINDOW: Duration = Duration::from_secs(30);

/// A Whisper model loaded on one device.
pub struct WhisperDecoder {
    state: WhisperState,
    /// End of text; every token id from here on is a special or timestamp token
    eot: i32,
    threads: usize,
}

impl WhisperDecoder {
    pub const fn new(state: WhisperState, eot: i32, threads: usize) -> Self {
        Self {
            state,
            eot,
            threads,
        }
    }

    fn detect_language(&mut self, pcm: &[f32]) -> Result<LanguageEstimate, WhisperEngineError> {
        self.state.pcm_to_mel(pcm, self.threads)?;
        let (id, probabilities) = self.state.lang_detect(0, self.threads)?;
        let tag = detected_tag(id)?;
        let probability = usize::try_from(id)
            .ok()
            .and_then(|i| probabilities.get(i).copied())
            .ok_or(WhisperEngineError::UnknownLanguageId { id })?;
        Ok(LanguageEstimate {
            tag,
            confidence: Some(probability),
        })
    }
}

unsafe extern "C" fn cancelled(token: *mut c_void) -> bool {
    // SAFETY: the pointer is the `&CancelToken` of the request, which outlives `full()`.
    unsafe { &*token.cast_const().cast::<CancelToken>() }.is_cancelled()
}

impl WindowDecoder for WhisperDecoder {
    fn limits(&self) -> WindowLimits {
        WindowLimits {
            input: InputLimit::Unbounded,
            window: ENCODER_WINDOW,
        }
    }

    fn supports(&self, language: &LanguageTag) -> bool {
        whisper_code(language).is_some()
    }

    fn decode(&mut self, request: DecodeRequest<'_>) -> Result<Decoded, DecodeError> {
        if request.cancel.is_cancelled() {
            return Ok(Decoded {
                hypothesis: Hypothesis::empty(Segmentation::Native),
                completion: Completion::Cancelled,
            });
        }
        let (language, estimate) = match request.language {
            LanguageRequest::Use(tag) => (tag.clone(), None),
            LanguageRequest::Detect => {
                let estimate = self.detect_language(request.pcm)?;
                (estimate.tag.clone(), Some(estimate))
            }
        };
        let code =
            whisper_code(&language).ok_or(DecodeError::UnsupportedLanguage(language.clone()))?;

        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: BEAM_SIZE,
            patience: -1.0,
        });
        params.set_language(Some(code));
        params.set_n_threads(i32::try_from(self.threads).unwrap_or(i32::MAX));
        params.set_print_timestamps(false);
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_no_context(true);
        params.set_token_timestamps(request.timing.times_words());
        // SAFETY: `cancelled` reads the token through the pointer only while `full()` runs, and
        // the request borrows the token for longer than that.
        unsafe {
            params.set_abort_callback(Some(cancelled));
            params.set_abort_callback_user_data(
                std::ptr::from_ref(request.cancel)
                    .cast_mut()
                    .cast::<c_void>(),
            );
        }

        let outcome = self.state.full(params, request.pcm);
        let completion = match outcome {
            Ok(()) => Completion::Complete,
            Err(_) if request.cancel.is_cancelled() => Completion::Cancelled,
            Err(error) => return Err(WhisperEngineError::from(error).into()),
        };
        Ok(Decoded {
            hypothesis: Hypothesis {
                segments: segments::read(&self.state, self.eot)?,
                language: estimate,
                segmentation: Segmentation::Native,
            },
            completion,
        })
    }
}
