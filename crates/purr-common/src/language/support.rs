use super::tag::LanguageTag;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The spoken languages a model transcribes.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LanguageSupport {
    Any,
    Only(Arc<[LanguageTag]>),
}

impl LanguageSupport {
    #[must_use]
    pub fn english_only() -> Self {
        Self::Only(Arc::from([LanguageTag::english()]))
    }

    #[must_use]
    pub fn is_any(&self) -> bool {
        match self {
            Self::Any => true,
            Self::Only(_) => false,
        }
    }

    /// Whether `tag` is a language the model transcribes.
    ///
    /// A tag matches a listed one when they have the same primary language and either has no
    /// script or region, or both are the same: `pt` is supported by `pt-BR` and `pt-BR` by `pt`,
    /// `pt-BR` is not supported by `pt-PT`.
    #[must_use]
    pub fn supports(&self, tag: &LanguageTag) -> bool {
        match self {
            Self::Any => true,
            Self::Only(_) => self.listed(tag).is_some(),
        }
    }

    /// The listed language `tag` matches, by the rule of [`Self::supports`]; `None` when the
    /// model takes any language, as none is listed.
    #[must_use]
    pub fn listed(&self, tag: &LanguageTag) -> Option<&LanguageTag> {
        match self {
            Self::Any => None,
            Self::Only(listed) => listed.iter().find(|listed| matches(listed, tag)),
        }
    }

    /// The one language the model transcribes, when it knows only one.
    #[must_use]
    pub fn single(&self) -> Option<&LanguageTag> {
        match self {
            Self::Any => None,
            Self::Only(listed) => match listed.as_ref() {
                [only] => Some(only),
                [] | [_, _, ..] => None,
            },
        }
    }
}

fn matches(listed: &LanguageTag, tag: &LanguageTag) -> bool {
    listed.primary() == tag.primary() && (listed.is_bare() || tag.is_bare() || listed == tag)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(text: &str) -> LanguageTag {
        text.parse().unwrap()
    }

    fn only(tags: &[&str]) -> LanguageSupport {
        LanguageSupport::Only(tags.iter().map(|text| tag(text)).collect())
    }

    #[test]
    fn any_supports_every_language() {
        assert!(LanguageSupport::Any.supports(&tag("yue")));
    }

    #[test]
    fn the_listed_language_is_the_one_the_tag_matches() {
        let support = only(&["en-US", "fr", "pt-BR"]);
        assert_eq!(support.listed(&tag("fr-CH")), Some(&tag("fr")));
        assert_eq!(support.listed(&tag("en")), Some(&tag("en-US")));
        assert_eq!(support.listed(&tag("pt-PT")), None);
        assert_eq!(LanguageSupport::Any.listed(&tag("fr")), None);
    }

    #[test]
    fn a_bare_tag_matches_a_region_qualified_list_by_primary() {
        let support = only(&["en-US", "pt-BR"]);
        assert!(support.supports(&tag("en")));
        assert!(support.supports(&tag("pt")));
        assert!(!support.supports(&tag("fr")));
    }

    #[test]
    fn a_qualified_tag_matches_its_own_region_or_a_bare_entry() {
        let support = only(&["en-US", "pt"]);
        assert!(support.supports(&tag("en-US")));
        assert!(!support.supports(&tag("en-GB")));
        assert!(support.supports(&tag("pt-BR")));
        assert!(support.supports(&tag("pt-PT")));
    }

    #[test]
    fn legacy_spellings_match_after_parsing() {
        assert!(only(&["he"]).supports(&tag("iw")));
        assert!(only(&["jv"]).supports(&tag("jw")));
    }

    #[test]
    fn support_is_stored_as_any_or_a_list() {
        assert_eq!(
            serde_json::to_string(&LanguageSupport::Any).unwrap(),
            "\"any\""
        );
        assert_eq!(
            serde_json::to_string(&LanguageSupport::english_only()).unwrap(),
            "{\"only\":[\"en\"]}"
        );
        assert_eq!(
            serde_json::from_str::<LanguageSupport>("{\"only\":[\"iw\",\"pt_br\"]}").unwrap(),
            only(&["he", "pt-BR"])
        );
    }

    #[test]
    fn english_only_knows_a_single_language() {
        let support = LanguageSupport::english_only();
        assert_eq!(support.single(), Some(&tag("en")));
        assert!(support.supports(&tag("en-GB")));
        assert!(!support.supports(&tag("fr")));
        assert_eq!(only(&["en", "fr"]).single(), None);
        assert_eq!(LanguageSupport::Any.single(), None);
    }
}
