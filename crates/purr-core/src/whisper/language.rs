//! The language a streaming transcription decodes in.

use crate::{Result, error::TranscriptionError};
use purr_common::language::{tag::LanguageTag, whisper};

/// Whisper's own spelling of `tag`, when whisper.cpp knows the language.
pub fn whisper_code(tag: &LanguageTag) -> Result<&'static str> {
    whisper_rs::get_lang_id(whisper::code(tag))
        .and_then(whisper_rs::get_lang_str)
        .ok_or_else(|| {
            TranscriptionError::UnknownLanguage {
                code: tag.to_string(),
            }
            .into()
        })
}

/// The tag of the language whisper.cpp reports under `id`.
pub fn detected_tag(id: i32) -> Result<LanguageTag> {
    whisper_rs::get_lang_str(id)
        .and_then(|code| whisper::tag(code).ok())
        .ok_or_else(|| TranscriptionError::UnknownLanguageId { id }.into())
}

/// How the language of a stream is settled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamLanguage {
    /// The user picked it
    Chosen(LanguageTag),
    /// Not known yet: each window is detected on its own
    Detecting,
    /// Detected confidently on a window with speech, and used for the rest of the stream
    Locked(LanguageTag),
}

impl StreamLanguage {
    /// The least probability Whisper's language detection must give before the stream adopts it
    pub const LOCK_PROBABILITY: f32 = 0.8;

    /// Folds one window's detection in; returns the language when this window locked it.
    pub fn observe(
        &mut self,
        detected: &LanguageTag,
        probability: f32,
        spoke: bool,
    ) -> Option<LanguageTag> {
        match self {
            Self::Detecting if spoke && probability >= Self::LOCK_PROBABILITY => {
                *self = Self::Locked(detected.clone());
                Some(detected.clone())
            }
            Self::Chosen(_) | Self::Detecting | Self::Locked(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(text: &str) -> LanguageTag {
        text.parse().unwrap()
    }

    #[test]
    fn locks_on_the_first_confident_window_with_speech() {
        let mut language = StreamLanguage::Detecting;
        assert_eq!(language.observe(&tag("ja"), 0.9, false), None);
        assert_eq!(language.observe(&tag("en"), 0.3, true), None);
        assert_eq!(language.observe(&tag("en"), 0.8, true), Some(tag("en")));
        assert_eq!(language, StreamLanguage::Locked(tag("en")));
        assert_eq!(language.observe(&tag("de"), 0.99, true), None);
        assert_eq!(language, StreamLanguage::Locked(tag("en")));
    }

    #[test]
    fn noise_detected_as_a_language_does_not_lock_it() {
        let mut language = StreamLanguage::Detecting;
        assert_eq!(language.observe(&tag("nn"), 0.452, true), None);
        assert_eq!(language.observe(&tag("en"), 0.363, true), None);
        assert_eq!(language, StreamLanguage::Detecting);
        assert_eq!(language.observe(&tag("en"), 0.943, true), Some(tag("en")));
    }

    #[test]
    fn a_chosen_language_never_locks_another() {
        let mut language = StreamLanguage::Chosen(tag("en"));
        assert_eq!(language.observe(&tag("fr"), 0.99, true), None);
        assert_eq!(language, StreamLanguage::Chosen(tag("en")));
    }

    #[test]
    fn every_language_whisper_reports_is_a_tag_that_maps_back() {
        for id in 0..=whisper_rs::get_lang_max_id() {
            let code = whisper_rs::get_lang_str(id).unwrap();
            let detected = detected_tag(id).unwrap();
            assert_eq!(whisper_code(&detected).unwrap(), code);
        }
    }

    #[test]
    fn every_listed_language_is_one_whisper_knows() {
        for known in purr_common::language::known::all() {
            assert!(whisper_code(&known.tag).is_ok(), "{}", known.tag);
        }
    }

    #[test]
    fn hebrew_and_javanese_reach_whisper_under_its_codes() {
        assert_eq!(whisper_code(&tag("he")).unwrap(), "he");
        assert_eq!(whisper_code(&tag("iw")).unwrap(), "he");
        assert_eq!(whisper_code(&tag("jv")).unwrap(), "jw");
        assert_eq!(whisper_code(&tag("pt-BR")).unwrap(), "pt");
        assert!(whisper_code(&tag("xx")).is_err());
    }
}
