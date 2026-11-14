use super::{known, support::LanguageSupport, tag::LanguageTag};

/// What the user asked for as the spoken language.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LanguageChoice {
    Automatic,
    Spoken(LanguageTag),
}

/// What the `ui.language` setting holds for [`LanguageChoice::Automatic`].
const AUTOMATIC: &str = "auto";

impl LanguageChoice {
    /// Reads the `ui.language` setting; text that is no language means detection.
    #[must_use]
    pub fn from_setting(text: &str) -> Self {
        if text.trim().eq_ignore_ascii_case(AUTOMATIC) {
            return Self::Automatic;
        }
        LanguageTag::parse(text).map_or(Self::Automatic, Self::Spoken)
    }

    /// The value to store in the `ui.language` setting.
    #[must_use]
    pub fn setting(&self) -> String {
        match self {
            Self::Automatic => AUTOMATIC.to_owned(),
            Self::Spoken(tag) => tag.as_str().to_owned(),
        }
    }

    /// The name shown in the pill and the list.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Automatic => "Automatic".to_owned(),
            Self::Spoken(tag) => known::display_name(tag).unwrap_or_else(|| tag.to_string()),
        }
    }

    /// What applies with a model that transcribes only some languages: a model that knows a
    /// single language transcribes it whatever was asked, and a listed language is kept.
    #[must_use]
    pub fn for_support(self, support: &LanguageSupport) -> Self {
        match (&self, support) {
            (_, LanguageSupport::Any) => self,
            (Self::Spoken(tag), LanguageSupport::Only(_)) if support.supports(tag) => self,
            (Self::Automatic | Self::Spoken(_), LanguageSupport::Only(listed)) => {
                match listed.as_ref() {
                    [only] => Self::Spoken(only.clone()),
                    [] | [_, _, ..] => self,
                }
            }
        }
    }

    /// The language to send with a transcription request; `None` asks the platform to detect it.
    #[must_use]
    pub fn request_tag(&self) -> Option<LanguageTag> {
        match self {
            Self::Automatic => None,
            Self::Spoken(tag) => Some(tag.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spoken(text: &str) -> LanguageChoice {
        LanguageChoice::Spoken(text.parse().unwrap())
    }

    #[test]
    fn the_setting_round_trips() {
        assert_eq!(
            LanguageChoice::from_setting("auto"),
            LanguageChoice::Automatic
        );
        assert_eq!(LanguageChoice::from_setting("fr"), spoken("fr"));
        assert_eq!(spoken("fr").setting(), "fr");
        assert_eq!(LanguageChoice::Automatic.setting(), "auto");
    }

    #[test]
    fn old_spellings_are_read_and_written_canonical() {
        assert_eq!(LanguageChoice::from_setting("iw"), spoken("he"));
        assert_eq!(LanguageChoice::from_setting("JW"), spoken("jv"));
        assert_eq!(LanguageChoice::from_setting("Fr"), spoken("fr"));
        assert_eq!(LanguageChoice::from_setting("pt_br"), spoken("pt-BR"));
        assert_eq!(LanguageChoice::from_setting("iw").setting(), "he");
        assert_eq!(
            LanguageChoice::from_setting("AUTO"),
            LanguageChoice::Automatic
        );
    }

    #[test]
    fn text_that_is_no_language_means_detection() {
        assert_eq!(LanguageChoice::from_setting(""), LanguageChoice::Automatic);
        assert_eq!(
            LanguageChoice::from_setting("klingon"),
            LanguageChoice::Automatic
        );
    }

    #[test]
    fn labels_are_capitalized_names() {
        assert_eq!(spoken("de").label(), "German");
        assert_eq!(spoken("iw").label(), "Hebrew");
        assert_eq!(spoken("fil").label(), "fil");
        assert_eq!(LanguageChoice::Automatic.label(), "Automatic");
    }

    #[test]
    fn a_model_that_knows_one_language_forces_it_in_the_request() {
        let english = LanguageSupport::english_only();
        assert_eq!(
            LanguageChoice::Automatic
                .for_support(&english)
                .request_tag(),
            Some(LanguageTag::english())
        );
        assert_eq!(
            spoken("fr").for_support(&english).request_tag(),
            Some(LanguageTag::english())
        );
        assert_eq!(
            LanguageChoice::Automatic
                .for_support(&LanguageSupport::Any)
                .request_tag(),
            None
        );
        assert_eq!(
            spoken("fr")
                .for_support(&LanguageSupport::Any)
                .request_tag(),
            Some("fr".parse().unwrap())
        );
    }

    #[test]
    fn a_model_that_knows_several_languages_keeps_a_listed_choice() {
        let support = LanguageSupport::Only(
            ["en-US", "pt-BR"]
                .iter()
                .map(|text| text.parse().unwrap())
                .collect(),
        );
        assert_eq!(spoken("pt").for_support(&support), spoken("pt"));
        assert_eq!(
            LanguageChoice::Automatic.for_support(&support),
            LanguageChoice::Automatic
        );
    }
}
