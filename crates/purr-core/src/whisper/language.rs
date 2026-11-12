//! The language a streaming transcription decodes in.

/// Whisper language code, such as `en`
pub type LanguageCode = &'static str;

/// How the language of a stream is settled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamLanguage {
    /// The user picked it
    Chosen(LanguageCode),
    /// Not known yet: each window is detected on its own
    Detecting,
    /// Detected confidently on a window with speech, and used for the rest of the stream
    Locked(LanguageCode),
}

impl StreamLanguage {
    /// The least probability Whisper's language detection must give before the stream adopts it
    pub const LOCK_PROBABILITY: f32 = 0.8;

    /// Folds one window's detection in; returns the language when this window locked it.
    pub fn observe(
        &mut self,
        detected: LanguageCode,
        probability: f32,
        spoke: bool,
    ) -> Option<LanguageCode> {
        match self {
            Self::Detecting if spoke && probability >= Self::LOCK_PROBABILITY => {
                *self = Self::Locked(detected);
                Some(detected)
            }
            Self::Chosen(_) | Self::Detecting | Self::Locked(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_on_the_first_confident_window_with_speech() {
        let mut language = StreamLanguage::Detecting;
        assert_eq!(language.observe("ja", 0.9, false), None);
        assert_eq!(language.observe("en", 0.3, true), None);
        assert_eq!(language.observe("en", 0.8, true), Some("en"));
        assert_eq!(language, StreamLanguage::Locked("en"));
        assert_eq!(language.observe("de", 0.99, true), None);
        assert_eq!(language, StreamLanguage::Locked("en"));
    }

    #[test]
    fn noise_detected_as_a_language_does_not_lock_it() {
        let mut language = StreamLanguage::Detecting;
        assert_eq!(language.observe("nn", 0.452, true), None);
        assert_eq!(language.observe("en", 0.363, true), None);
        assert_eq!(language, StreamLanguage::Detecting);
        assert_eq!(language.observe("en", 0.943, true), Some("en"));
    }

    #[test]
    fn a_chosen_language_never_locks_another() {
        let mut language = StreamLanguage::Chosen("en");
        assert_eq!(language.observe("fr", 0.99, true), None);
        assert_eq!(language, StreamLanguage::Chosen("en"));
    }
}
