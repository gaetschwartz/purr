use super::tag::LanguageTag;
use std::sync::Arc;

/// The spoken languages a model transcribes.
#[derive(Clone, Debug, PartialEq, Eq)]
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
            Self::Only(listed) => listed.iter().any(|listed| matches(listed, tag)),
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
    fn english_only_knows_a_single_language() {
        let support = LanguageSupport::english_only();
        assert_eq!(support.single(), Some(&tag("en")));
        assert!(support.supports(&tag("en-GB")));
        assert!(!support.supports(&tag("fr")));
        assert_eq!(only(&["en", "fr"]).single(), None);
        assert_eq!(LanguageSupport::Any.single(), None);
    }
}
