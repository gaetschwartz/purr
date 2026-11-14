//! The language a streaming transcription decodes in.

use purr_common::language::tag::LanguageTag;

use crate::engine::decoder::LanguageRequest;

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
    /// The least confidence a detection must have before the stream adopts it
    pub const LOCK_CONFIDENCE: f32 = 0.8;

    /// What the next decode is asked to do about the language.
    #[must_use]
    pub fn request(&self) -> LanguageRequest {
        match self {
            Self::Chosen(tag) | Self::Locked(tag) => LanguageRequest::Use(tag.clone()),
            Self::Detecting => LanguageRequest::Detect,
        }
    }

    /// Folds one window's detection in; returns the language when this window locked it.
    ///
    /// An engine that does not say how sure it is stands by its answer, so its first detection
    /// on a window with speech locks the language.
    pub fn observe(
        &mut self,
        detected: &LanguageTag,
        confidence: Option<f32>,
        spoke: bool,
    ) -> Option<LanguageTag> {
        match self {
            Self::Detecting if spoke && confidence.is_none_or(|p| p >= Self::LOCK_CONFIDENCE) => {
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
        assert_eq!(language.observe(&tag("ja"), Some(0.9), false), None);
        assert_eq!(language.observe(&tag("en"), Some(0.3), true), None);
        assert_eq!(
            language.observe(&tag("en"), Some(0.8), true),
            Some(tag("en"))
        );
        assert_eq!(language, StreamLanguage::Locked(tag("en")));
        assert_eq!(language.observe(&tag("de"), Some(0.99), true), None);
        assert_eq!(language, StreamLanguage::Locked(tag("en")));
    }

    #[test]
    fn noise_detected_as_a_language_does_not_lock_it() {
        let mut language = StreamLanguage::Detecting;
        assert_eq!(language.observe(&tag("nn"), Some(0.452), true), None);
        assert_eq!(language.observe(&tag("en"), Some(0.363), true), None);
        assert_eq!(language, StreamLanguage::Detecting);
        assert_eq!(
            language.observe(&tag("en"), Some(0.943), true),
            Some(tag("en"))
        );
    }

    #[test]
    fn a_chosen_language_never_locks_another() {
        let mut language = StreamLanguage::Chosen(tag("en"));
        assert_eq!(language.observe(&tag("fr"), Some(0.99), true), None);
        assert_eq!(language, StreamLanguage::Chosen(tag("en")));
    }

    #[test]
    fn a_detection_without_confidence_locks_on_the_first_window_with_speech() {
        let mut language = StreamLanguage::Detecting;
        assert_eq!(language.observe(&tag("fr"), None, false), None);
        assert_eq!(language.observe(&tag("fr"), None, true), Some(tag("fr")));
        assert_eq!(language, StreamLanguage::Locked(tag("fr")));
    }

    #[test]
    fn the_request_follows_the_state() {
        assert_eq!(StreamLanguage::Detecting.request(), LanguageRequest::Detect);
        assert_eq!(
            StreamLanguage::Locked(tag("de")).request(),
            LanguageRequest::Use(tag("de"))
        );
    }
}
