pub mod platform;
pub mod settings;

macro_rules! make_languages_enum {
    (
      LANGUAGES = { $($code:literal : $lang:literal,)* $(,)? },
      TO_LANGUAGE_CODE = { $($lang2:literal : $code2:literal,)* $(,)?} $(,)?
    ) => { paste::paste! {
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub enum Language {
            #[doc = "Language: `auto` (Special value for auto-detect)"]
            Auto,
            $(
                #[doc = "Language: `"  $lang "` (`" $code "`)"]
                [<$code:camel>],
            )*
        }

        #[allow(dead_code)]
        impl Language {
            pub fn code(&self) -> &'static str {
                match self {
                    Language::Auto => "auto",  // special case for auto-detect
                    $(Language::[<$code:camel>] => $code, )*
                }
            }

            pub fn name(&self) -> &'static str {
                match self {
                    Language::Auto => "auto",  // special case for auto-detect
                    $(Language::[<$code:camel>] => $lang, )*
                }
            }

            pub const VARIANTS_CODES: &'static [&'static str] = &[
                "auto",
                $($code,)*
            ];

            pub const VARIANTS_LANGUAGES: &'static [&'static str] = &[
                "auto",
                $($lang,)*
            ];


            pub const ADDITIONAL_MAPPINGS: &'static [(&'static str, &'static str)] = &[
                $(($lang2, $code2),)*
            ];

            // phf map
            pub const MAPPINGS: phf::Map<&'static str, Language> = phf::phf_map! {
                "auto" => Language::Auto,
                $($code => Language::[<$code:camel>],)*
                $($lang => Language::[<$code:camel>],)*
                $($lang2 => Language::[<$code2:camel>],)*
            };

            pub const VARIANTS: &'static [Language] = &[
                Language::Auto,
                $(Language::[<$code:camel>],)*
            ];
        }

        #[cfg(feature = "clap")]
        impl clap::ValueEnum for Language {
            fn value_variants<'a>() -> &'a [Self] {
                Self::VARIANTS
            }

            fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
                const MAX_ADDITIONAL_ALIASES: usize = 2;
                const MAX_TOTAL_ALIASES: usize = 1 + MAX_ADDITIONAL_ALIASES;
                let my_code = self.code();
                let mut aliases = Vec::with_capacity(MAX_TOTAL_ALIASES);
                aliases.push(self.name());
                for (lang, code) in Language::ADDITIONAL_MAPPINGS {
                    if *code == my_code {
                        aliases.push(*lang);
                        if aliases.len() >= MAX_TOTAL_ALIASES {
                            break;
                        }
                    }
                }
                Some(
                  clap::builder::PossibleValue::new(self.code())
                    .aliases(aliases)
                )
            }
        }

        impl std::str::FromStr for Language {
            type Err = ();

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Self::MAPPINGS.get(s).cloned().ok_or(())
            }
        }
    } };
}

// https://github.com/openai/whisper/blob/8cf36f3508c9acd341a45eb2364239a3d81458b9/whisper/tokenizer.py#L10-L110
make_languages_enum!(LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "iw": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}, TO_LANGUAGE_CODE = {
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
});
