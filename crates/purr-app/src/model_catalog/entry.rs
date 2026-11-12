use super::family::{ModelFamily, ModelGroup};
use purr_common::platform::{ModelInfo, model_flag};

/// Platform attribute that names the model's size class.
const FAMILY_ATTRIBUTE: &str = "family";

/// The languages a model transcribes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Multilingual,
    EnglishOnly,
}

impl Language {
    /// The tag text.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Multilingual => "Multilingual",
            Self::EnglishOnly => "English only",
        }
    }
}

/// How the weights are stored; quantized ones are smaller and slightly less accurate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Precision {
    Full,
    Quantized {
        bits: u8,
    },
    /// Encoder and decoder use different precisions, as the browser models do.
    Mixed,
    /// A precision this version of Purr does not know, as the platform spelled it.
    Other(String),
}

impl Precision {
    /// Reads the platform's quantization attribute: `fp16`, `q5_1`, or `encoder/decoder` pairs.
    #[must_use]
    pub fn parse(raw: Option<&str>) -> Self {
        let Some(raw) = raw else {
            return Self::Full;
        };
        match raw.split_once('/') {
            Some((encoder, decoder)) => {
                if Self::parse_single(encoder) == Self::Full
                    && Self::parse_single(decoder) == Self::Full
                {
                    Self::Full
                } else {
                    Self::Mixed
                }
            }
            None => Self::parse_single(raw),
        }
    }

    fn parse_single(raw: &str) -> Self {
        match raw {
            "fp32" | "fp16" => Self::Full,
            "int8" | "uint8" => Self::Quantized { bits: 8 },
            other => {
                let bits = other
                    .strip_prefix('q')
                    .map(|rest| {
                        rest.chars()
                            .take_while(char::is_ascii_digit)
                            .collect::<String>()
                    })
                    .and_then(|digits| digits.parse::<u8>().ok());
                bits.map_or_else(
                    || Self::Other(other.to_owned()),
                    |bits| Self::Quantized { bits },
                )
            }
        }
    }

    /// The tag text, absent for full precision and for the mixed precision every browser model has.
    #[must_use]
    pub fn tag(&self) -> Option<String> {
        match self {
            Self::Full | Self::Mixed => None,
            Self::Quantized { bits } => Some(format!("Quantized Q{bits}")),
            Self::Other(raw) => Some(raw.clone()),
        }
    }
}

/// A model of the catalog as the Models screen and the pickers describe it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEntry {
    pub id: String,
    /// The friendly name without the precision, such as `Tiny (English)`.
    pub name: String,
    /// The name with the precision, such as `Tiny (English) · Q5`.
    pub label: String,
    pub family: Option<ModelFamily>,
    pub group: ModelGroup,
    pub language: Language,
    pub precision: Precision,
    pub speaker_turns: bool,
    pub requires_shader_f16: bool,
    pub size_bytes: u64,
    pub installed: bool,
    description: String,
}

impl ModelEntry {
    /// Reads the platform's description of a model.
    #[must_use]
    pub fn from_info(info: &ModelInfo) -> Self {
        let metadata = &info.metadata;
        let family = metadata
            .platform_specific
            .get(FAMILY_ATTRIBUTE)
            .and_then(|raw| ModelFamily::parse(raw));
        let language = if metadata.has_flag(model_flag::ENGLISH_ONLY) {
            Language::EnglishOnly
        } else {
            Language::Multilingual
        };
        let precision = Precision::parse(metadata.quantization.as_deref());
        let speaker_turns = metadata.has_flag(model_flag::SPEAKER_TURNS);

        let mut name = family.map_or_else(|| info.name.clone(), |family| family.label().to_owned());
        if family.is_some() && language == Language::EnglishOnly {
            name.push_str(" (English)");
        }
        let mut label = name.clone();
        if let Precision::Quantized { bits } = precision {
            label = format!("{label} · Q{bits}");
        }
        if speaker_turns {
            label = format!("{label} · Speaker turns");
        }

        Self {
            id: info.id.clone(),
            name,
            label,
            family,
            group: family.map_or(ModelGroup::Other, ModelFamily::group),
            language,
            precision,
            speaker_turns,
            requires_shader_f16: metadata.has_flag(model_flag::REQUIRES_SHADER_F16),
            size_bytes: info.size_bytes,
            installed: info.is_installed,
            description: info.description.clone(),
        }
    }

    /// One plain sentence on what sets this variant apart, when its tags don't say it already.
    #[must_use]
    pub fn hint(&self) -> Option<String> {
        if let Some(note) = self.family.and_then(ModelFamily::note) {
            return Some(note.to_owned());
        }
        if self.speaker_turns {
            return Some("Marks where the speaker changes.".to_owned());
        }
        let described = Some(self.description.clone()).filter(|text| !text.trim().is_empty());
        if self.family.is_none() {
            return described;
        }
        match (&self.precision, self.language) {
            (Precision::Other(_), _) => described,
            (Precision::Quantized { bits }, _) => {
                let size = if *bits >= 8 { "half" } else { "a third" };
                Some(format!("About {size} of the size, slightly less accurate."))
            }
            (Precision::Mixed | Precision::Full, Language::EnglishOnly) => {
                Some("A little more accurate on English speech.".to_owned())
            }
            (Precision::Mixed | Precision::Full, Language::Multilingual) => None,
        }
    }

    /// Whether every word of `query` appears in the name, id or tags, ignoring case.
    #[must_use]
    pub fn matches(&self, query: &str) -> bool {
        let haystack = format!(
            "{} {} {} {}",
            self.label,
            self.id,
            self.language.label(),
            self.precision.tag().unwrap_or_default()
        )
        .to_lowercase();
        query
            .split_whitespace()
            .all(|word| haystack.contains(&word.to_lowercase()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::platform::ModelMetadata;

    fn info(id: &str, family: &str, quantization: &str, english: bool) -> ModelInfo {
        let metadata = ModelMetadata::new(
            "ggml".to_owned(),
            "whisper".to_owned(),
            "multilingual".to_owned(),
            "v1".to_owned(),
        )
        .with_quantization(quantization.to_owned())
        .with_platform_attribute(FAMILY_ATTRIBUTE.to_owned(), family.to_owned())
        .with_platform_attribute(model_flag::ENGLISH_ONLY.to_owned(), english.to_string());
        ModelInfo::new(
            id.to_owned(),
            id.to_owned(),
            "A model".to_owned(),
            1024,
            false,
            metadata,
        )
    }

    #[test]
    fn precision_reads_desktop_and_browser_spellings() {
        assert_eq!(Precision::parse(None), Precision::Full);
        assert_eq!(Precision::parse(Some("fp16")), Precision::Full);
        assert_eq!(
            Precision::parse(Some("q5_1")),
            Precision::Quantized { bits: 5 }
        );
        assert_eq!(
            Precision::parse(Some("q8_0")),
            Precision::Quantized { bits: 8 }
        );
        assert_eq!(Precision::parse(Some("fp32/q4")), Precision::Mixed);
        assert_eq!(Precision::parse(Some("fp32/fp16")), Precision::Full);
        assert_eq!(
            Precision::parse(Some("weird")),
            Precision::Other("weird".to_owned())
        );
    }

    #[test]
    fn a_quantized_english_model_gets_a_friendly_label() {
        let entry = ModelEntry::from_info(&info("tiny.en-q5_1", "tiny", "q5_1", true));
        assert_eq!(entry.name, "Tiny (English)");
        assert_eq!(entry.label, "Tiny (English) · Q5");
        assert_eq!(entry.group, ModelGroup::Tiny);
        assert_eq!(entry.language, Language::EnglishOnly);
    }

    #[test]
    fn web_family_spellings_group_with_their_size_class() {
        let entry = ModelEntry::from_info(&info("base.en", "base.en", "fp32/q4", true));
        assert_eq!(entry.family, Some(ModelFamily::Base));
        assert_eq!(entry.label, "Base (English)");
        assert_eq!(entry.precision, Precision::Mixed);
    }

    #[test]
    fn an_unknown_family_keeps_the_platform_name_and_lands_in_other() {
        let entry = ModelEntry::from_info(&info("mystery", "mystery", "fp16", false));
        assert_eq!(entry.name, "mystery");
        assert_eq!(entry.group, ModelGroup::Other);
        assert_eq!(entry.hint().as_deref(), Some("A model"));
    }

    #[test]
    fn hints_follow_the_variant() {
        let full = ModelEntry::from_info(&info("small", "small", "fp16", false));
        assert_eq!(full.hint(), None);
        let english = ModelEntry::from_info(&info("small.en", "small", "fp16", true));
        assert_eq!(
            english.hint().as_deref(),
            Some("A little more accurate on English speech.")
        );
        let quantized = ModelEntry::from_info(&info("small-q5_1", "small", "q5_1", false));
        assert_eq!(
            quantized.hint().as_deref(),
            Some("About a third of the size, slightly less accurate.")
        );
        let eight_bit = ModelEntry::from_info(&info("small-q8_0", "small", "q8_0", false));
        assert_eq!(
            eight_bit.hint().as_deref(),
            Some("About half of the size, slightly less accurate.")
        );
        let old = ModelEntry::from_info(&info("large-v1", "large-v1", "fp16", false));
        assert_eq!(
            old.hint().as_deref(),
            Some("First large model, superseded by v3.")
        );
    }

    #[test]
    fn search_needs_every_word_and_ignores_case() {
        let entry = ModelEntry::from_info(&info("small.en-q5_1", "small", "q5_1", true));
        assert!(entry.matches("SMALL english"));
        assert!(entry.matches("q5"));
        assert!(entry.matches(""));
        assert!(!entry.matches("small medium"));
    }
}
