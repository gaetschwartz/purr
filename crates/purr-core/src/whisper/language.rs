//! Languages as whisper.cpp names them.

use purr_common::language::{tag::LanguageTag, whisper};

use super::error::WhisperEngineError;

/// Whisper's own spelling of `tag`, when whisper.cpp knows the language.
pub fn whisper_code(tag: &LanguageTag) -> Option<&'static str> {
    whisper_rs::get_lang_id(whisper::code(tag)).and_then(whisper_rs::get_lang_str)
}

/// The tag of the language whisper.cpp reports under `id`.
pub fn detected_tag(id: i32) -> Result<LanguageTag, WhisperEngineError> {
    whisper_rs::get_lang_str(id)
        .and_then(|code| whisper::tag(code).ok())
        .ok_or(WhisperEngineError::UnknownLanguageId { id })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(text: &str) -> LanguageTag {
        text.parse().unwrap()
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
            assert!(whisper_code(&known.tag).is_some(), "{}", known.tag);
        }
    }

    #[test]
    fn hebrew_and_javanese_reach_whisper_under_its_codes() {
        assert_eq!(whisper_code(&tag("he")), Some("he"));
        assert_eq!(whisper_code(&tag("iw")), Some("he"));
        assert_eq!(whisper_code(&tag("jv")), Some("jw"));
        assert_eq!(whisper_code(&tag("pt-BR")), Some("pt"));
        assert_eq!(whisper_code(&tag("xx")), None);
    }
}
