use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{fmt, str::FromStr, sync::Arc};

/// A spoken language as an engine-independent BCP-47 tag in canonical form: a lowercase primary
/// language, then an optional Title-case script and an optional upper-case region, joined by `-`.
///
/// Engines spell languages their own way; they map from and to this type at their edge.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LanguageTag(Arc<str>);

#[oopsie::oopsie]
#[derive(Clone, PartialEq, Eq)]
pub enum LanguageTagError {
    #[oopsie("A language tag cannot be empty")]
    Empty,
    #[oopsie("`{tag}` does not start with a two or three letter language code")]
    InvalidLanguage { tag: String },
    #[oopsie("`{subtag}` in `{tag}` is not a script or a region")]
    InvalidSubtag { tag: String, subtag: String },
}

/// Language codes that were retired for another, and what replaced them.
const RETIRED: &[(&str, &str)] = &[("iw", "he"), ("jw", "jv"), ("in", "id"), ("ji", "yi")];

impl LanguageTag {
    pub fn parse(text: &str) -> Result<Self, LanguageTagError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(LanguageTagError::Empty);
        }
        let mut subtags = text.split(['-', '_']);
        let primary = subtags.next().unwrap_or_default().to_ascii_lowercase();
        let is_language = (2..=3).contains(&primary.len())
            && primary.bytes().all(|byte| byte.is_ascii_lowercase());
        if !is_language {
            return Err(LanguageTagError::InvalidLanguage {
                tag: text.to_owned(),
            });
        }
        let primary = RETIRED
            .iter()
            .find(|(retired, _)| *retired == primary)
            .map_or(primary.as_str(), |&(_, current)| current);
        let mut canonical = primary.to_owned();
        let mut script_allowed = true;
        let mut region_allowed = true;
        for subtag in subtags {
            let invalid = || LanguageTagError::InvalidSubtag {
                tag: text.to_owned(),
                subtag: subtag.to_owned(),
            };
            let is_script =
                subtag.len() == 4 && subtag.bytes().all(|byte| byte.is_ascii_alphabetic());
            let is_region = (subtag.len() == 2
                && subtag.bytes().all(|byte| byte.is_ascii_alphabetic()))
                || (subtag.len() == 3 && subtag.bytes().all(|byte| byte.is_ascii_digit()));
            if is_script && script_allowed {
                script_allowed = false;
                let mut letters = subtag.chars();
                let first = letters.next().ok_or_else(invalid)?;
                canonical.push('-');
                canonical.extend(first.to_uppercase());
                canonical.push_str(&letters.as_str().to_ascii_lowercase());
            } else if is_region && region_allowed {
                script_allowed = false;
                region_allowed = false;
                canonical.push('-');
                canonical.push_str(&subtag.to_ascii_uppercase());
            } else {
                return Err(invalid());
            }
        }
        Ok(Self(canonical.into()))
    }

    #[must_use]
    pub fn english() -> Self {
        Self(Arc::from("en"))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The language without script or region: `pt` for `pt-BR`.
    #[must_use]
    pub fn primary(&self) -> &str {
        self.0.split('-').next().unwrap_or_default()
    }

    /// Whether the tag is only a language, with no script or region.
    #[must_use]
    pub fn is_bare(&self) -> bool {
        !self.0.contains('-')
    }
}

impl FromStr for LanguageTag {
    type Err = LanguageTagError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::parse(text)
    }
}

impl fmt::Display for LanguageTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for LanguageTag {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for LanguageTag {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(text: &str) -> LanguageTag {
        text.parse().unwrap()
    }

    #[test]
    fn retired_codes_become_their_replacement() {
        assert_eq!(tag("iw").as_str(), "he");
        assert_eq!(tag("jw").as_str(), "jv");
        assert_eq!(tag("in").as_str(), "id");
        assert_eq!(tag("ji").as_str(), "yi");
        assert_eq!(tag("IW-il").as_str(), "he-IL");
    }

    #[test]
    fn spelling_is_normalised() {
        assert_eq!(tag("FR").as_str(), "fr");
        assert_eq!(tag(" Fr ").as_str(), "fr");
        assert_eq!(tag("en_US").as_str(), "en-US");
        assert_eq!(tag("EN-us").as_str(), "en-US");
        assert_eq!(tag("pt-br").as_str(), "pt-BR");
        assert_eq!(tag("zh-hans-cn").as_str(), "zh-Hans-CN");
        assert_eq!(tag("es-419").as_str(), "es-419");
        assert_eq!(tag("yue").as_str(), "yue");
    }

    #[test]
    fn normalising_twice_changes_nothing() {
        for text in ["iw", "en_us", "ZH-hant-tw", "jw", "fil"] {
            let once = tag(text);
            assert_eq!(tag(once.as_str()), once);
        }
    }

    #[test]
    fn invalid_tags_are_refused_with_the_reason() {
        assert_eq!(LanguageTag::parse(""), Err(LanguageTagError::Empty));
        assert_eq!(LanguageTag::parse("  "), Err(LanguageTagError::Empty));
        assert_eq!(
            LanguageTag::parse("auto"),
            Err(LanguageTagError::InvalidLanguage {
                tag: "auto".to_owned()
            })
        );
        assert_eq!(
            LanguageTag::parse("e"),
            Err(LanguageTagError::InvalidLanguage {
                tag: "e".to_owned()
            })
        );
        assert_eq!(
            LanguageTag::parse("1n"),
            Err(LanguageTagError::InvalidLanguage {
                tag: "1n".to_owned()
            })
        );
        assert_eq!(
            LanguageTag::parse("en-"),
            Err(LanguageTagError::InvalidSubtag {
                tag: "en-".to_owned(),
                subtag: String::new()
            })
        );
        assert_eq!(
            LanguageTag::parse("en-US-GB"),
            Err(LanguageTagError::InvalidSubtag {
                tag: "en-US-GB".to_owned(),
                subtag: "GB".to_owned()
            })
        );
        assert_eq!(
            LanguageTag::parse("en-US-Latn"),
            Err(LanguageTagError::InvalidSubtag {
                tag: "en-US-Latn".to_owned(),
                subtag: "Latn".to_owned()
            })
        );
        assert_eq!(
            LanguageTag::parse("french"),
            Err(LanguageTagError::InvalidLanguage {
                tag: "french".to_owned()
            })
        );
    }

    #[test]
    fn the_primary_subtag_drops_script_and_region() {
        assert_eq!(tag("pt-BR").primary(), "pt");
        assert_eq!(tag("zh-Hans-CN").primary(), "zh");
        assert_eq!(tag("fr").primary(), "fr");
        assert!(tag("fr").is_bare());
        assert!(!tag("fr-CA").is_bare());
    }

    #[test]
    fn a_tag_is_stored_as_its_canonical_string() {
        let stored = serde_json::to_string(&tag("pt_br")).unwrap();
        assert_eq!(stored, "\"pt-BR\"");
        let loaded: LanguageTag = serde_json::from_str("\"iw\"").unwrap();
        assert_eq!(loaded, tag("he"));
        assert!(serde_json::from_str::<LanguageTag>("\"auto\"").is_err());
    }
}
