use super::tag::LanguageTag;
use std::sync::LazyLock;

/// A language the pickers list, with the name it is shown and searched by.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KnownLanguage {
    pub tag: LanguageTag,
    pub name: &'static str,
}

/// In the order the pickers list them.
const NAMES: &[(&str, &str)] = &[
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jv", "javanese"),
    ("su", "sundanese"),
];

/// Languages that models other than Whisper transcribe: they are named and found, but are not
/// offered to a model that takes any language, which is Whisper's list.
const ALSO_NAMED: &[(&str, &str)] = &[("yue", "cantonese"), ("fil", "filipino")];

/// Other names that find a language, by the code of the language.
const ALIASES: &[(&str, &str)] = &[
    ("burmese", "my"),
    ("valencian", "ca"),
    ("flemish", "nl"),
    ("haitian", "ht"),
    ("letzeburgesch", "lb"),
    ("pushto", "ps"),
    ("panjabi", "pa"),
    ("moldavian", "ro"),
    ("moldovan", "ro"),
    ("sinhalese", "si"),
    ("castilian", "es"),
];

fn known_languages(names: &[(&'static str, &'static str)]) -> Vec<KnownLanguage> {
    names
        .iter()
        .filter_map(|&(code, name)| {
            LanguageTag::parse(code)
                .ok()
                .map(|tag| KnownLanguage { tag, name })
        })
        .collect()
}

static KNOWN: LazyLock<Vec<KnownLanguage>> = LazyLock::new(|| known_languages(NAMES));

static ALSO_KNOWN: LazyLock<Vec<KnownLanguage>> = LazyLock::new(|| known_languages(ALSO_NAMED));

fn named() -> impl Iterator<Item = &'static KnownLanguage> {
    KNOWN.iter().chain(ALSO_KNOWN.iter())
}

/// The languages the pickers list, in order.
#[must_use]
pub fn all() -> &'static [KnownLanguage] {
    &KNOWN
}

/// The language that goes by `query`: its code, its name or another name for it, in any case.
#[must_use]
pub fn find(query: &str) -> Option<&'static KnownLanguage> {
    let query = query.trim().to_lowercase();
    let by_name = named().find(|known| known.name == query);
    let by_alias = || {
        ALIASES
            .iter()
            .find(|(alias, _)| *alias == query)
            .and_then(|(_, code)| named().find(|known| known.tag.as_str() == *code))
    };
    let by_tag = || {
        LanguageTag::parse(&query)
            .ok()
            .and_then(|tag| named().find(|known| known.tag == tag))
    };
    by_name.or_else(by_alias).or_else(by_tag)
}

/// The name `tag` is shown by, upper-cased on its first letter; a region-qualified tag is shown
/// by the name of its language when it is not listed itself.
#[must_use]
pub fn display_name(tag: &LanguageTag) -> Option<String> {
    let exact = named().find(|known| known.tag == *tag);
    let by_language = || named().find(|known| known.tag.as_str() == tag.primary());
    exact
        .or_else(by_language)
        .map(|known| capitalized(known.name))
}

fn capitalized(word: &str) -> String {
    let mut chars = word.chars();
    chars
        .next()
        .map(|first| first.to_uppercase().chain(chars).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn tag(text: &str) -> LanguageTag {
        text.parse().unwrap()
    }

    #[test]
    fn every_listed_language_has_a_tag_and_none_is_listed_twice() {
        assert_eq!(all().len(), NAMES.len());
        let tags: HashSet<_> = all().iter().map(|known| known.tag.clone()).collect();
        assert_eq!(tags.len(), all().len());
        let names: HashSet<_> = all().iter().map(|known| known.name).collect();
        assert_eq!(names.len(), all().len());
    }

    #[test]
    fn the_list_is_in_whisper_order() {
        let first: Vec<_> = all()
            .iter()
            .take(3)
            .map(|known| known.tag.as_str())
            .collect();
        assert_eq!(first, ["en", "zh", "de"]);
    }

    #[test]
    fn hebrew_and_javanese_are_listed_by_their_current_codes() {
        assert_eq!(display_name(&tag("he")).as_deref(), Some("Hebrew"));
        assert_eq!(display_name(&tag("iw")).as_deref(), Some("Hebrew"));
        assert_eq!(display_name(&tag("jv")).as_deref(), Some("Javanese"));
        assert_eq!(display_name(&tag("jw")).as_deref(), Some("Javanese"));
    }

    #[test]
    fn languages_are_found_by_code_name_or_other_name() {
        assert_eq!(find("fr").map(|known| known.name), Some("french"));
        assert_eq!(find("French").map(|known| known.name), Some("french"));
        assert_eq!(find("flemish").map(|known| known.name), Some("dutch"));
        assert_eq!(find("Castilian").map(|known| known.name), Some("spanish"));
        assert_eq!(find("iw").map(|known| known.name), Some("hebrew"));
        assert_eq!(find("pt_br").map(|known| known.name), None);
        assert_eq!(find("zzz"), None);
    }

    #[test]
    fn a_regional_tag_is_named_after_its_language() {
        assert_eq!(display_name(&tag("pt-BR")).as_deref(), Some("Portuguese"));
        assert_eq!(display_name(&tag("zzz")), None);
    }

    #[test]
    fn languages_only_other_models_transcribe_are_named_but_not_offered_to_any_model() {
        assert_eq!(display_name(&tag("yue")).as_deref(), Some("Cantonese"));
        assert_eq!(display_name(&tag("fil")).as_deref(), Some("Filipino"));
        assert_eq!(
            find("Cantonese").map(|known| known.tag.as_str()),
            Some("yue")
        );
        assert!(all().iter().all(|known| known.tag != tag("yue")));
    }

    #[test]
    fn capitalizes_the_first_letter_only() {
        assert_eq!(capitalized("french"), "French");
        assert_eq!(capitalized("haitian creole"), "Haitian creole");
        assert_eq!(capitalized(""), "");
    }
}
