use super::family::{ModelGroup, ModelLine};
use crate::utils::counted::counted;
use purr_common::{
    language::{choice::LanguageChoice, support::LanguageSupport},
    model::{
        descriptor::{ModelDescriptor, Weights},
        family::ModelFamily,
        precision::Precision,
        profile::Speakers,
    },
    platform::{ModelInfo, ModelOrigin},
};

/// The tag text of the languages a model transcribes.
#[must_use]
pub fn languages_label(support: &LanguageSupport) -> String {
    match support {
        LanguageSupport::Any => "Multilingual".to_owned(),
        LanguageSupport::Only(tags) => match support.single() {
            Some(tag) => format!("{} only", LanguageChoice::Spoken(tag.clone()).label()),
            None => counted(tags.len(), "language", "languages"),
        },
    }
}

/// The tag text of how a model's weights are stored: the format's own name, with "Quantized" in
/// front when it is reduced precision. Browser weights keep [`precision_tag`].
#[must_use]
pub fn quantization_tag(weights: Weights) -> Option<String> {
    match weights {
        Weights::Quantized(quantization) => Some(match quantization.precision() {
            Precision::Quantized { .. } => format!("Quantized {}", quantization.label()),
            Precision::Full | Precision::Mixed => quantization.label().to_owned(),
        }),
        Weights::Onnx { .. } => precision_tag(weights.precision()),
    }
}

/// The tag text of a precision, absent for full precision and for the mixed precision every
/// browser model has.
#[must_use]
pub fn precision_tag(precision: Precision) -> Option<String> {
    match precision {
        Precision::Full | Precision::Mixed => None,
        Precision::Quantized { bits } => Some(format!("Quantized Q{bits}")),
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
    pub group: ModelGroup,
    pub descriptor: ModelDescriptor,
    pub size_bytes: u64,
    pub installed: bool,
    /// Whether the engine that runs the model is part of this build.
    pub engine_built: bool,
    pub origin: ModelOrigin,
}

impl ModelEntry {
    /// Reads the platform's description of a model.
    #[must_use]
    pub fn from_info(info: &ModelInfo) -> Self {
        let descriptor = info.descriptor.clone();
        let family = descriptor.family;
        let english_only = descriptor.profile.languages == LanguageSupport::english_only();
        let speaker_turns = descriptor.profile.speakers == Speakers::TurnMarkers;

        let mut name = family.label().to_owned();
        if english_only {
            name.push_str(" (English)");
        }
        let mut label = name.clone();
        if let Precision::Quantized { bits } = descriptor.weights.precision() {
            label = format!("{label} · Q{bits}");
        }
        if speaker_turns {
            label = format!("{label} · Speaker turns");
        }

        Self {
            id: info.id.clone(),
            name,
            label,
            group: ModelGroup::of(family),
            descriptor,
            size_bytes: info.size_bytes,
            installed: info.is_installed,
            engine_built: info.engine_built,
            origin: info.origin,
        }
    }

    #[must_use]
    pub const fn family(&self) -> ModelFamily {
        self.descriptor.family
    }

    #[must_use]
    pub const fn languages(&self) -> &LanguageSupport {
        &self.descriptor.profile.languages
    }

    /// Whether the model transcribes more than one language.
    #[must_use]
    pub fn is_multilingual(&self) -> bool {
        match self.languages() {
            LanguageSupport::Any => true,
            LanguageSupport::Only(tags) => tags.len() > 1,
        }
    }

    #[must_use]
    pub const fn line(&self) -> ModelLine {
        ModelLine::of(self.descriptor.family)
    }

    /// The tag text of how the weights are stored.
    #[must_use]
    pub fn quantization_tag(&self) -> Option<String> {
        match self.origin {
            ModelOrigin::Developer => None,
            ModelOrigin::Catalog => quantization_tag(self.descriptor.weights),
        }
    }

    #[must_use]
    pub fn is_english_only(&self) -> bool {
        *self.languages() == LanguageSupport::english_only()
    }

    #[must_use]
    pub fn precision(&self) -> Precision {
        self.descriptor.weights.precision()
    }

    #[must_use]
    pub const fn speaker_turns(&self) -> bool {
        match self.descriptor.profile.speakers {
            Speakers::TurnMarkers => true,
            Speakers::None | Speakers::Labelled => false,
        }
    }

    #[must_use]
    pub const fn requires_shader_f16(&self) -> bool {
        self.descriptor.weights.requires_shader_f16()
    }

    /// The browser can run the model on WebGPU only, not on the CPU.
    #[must_use]
    pub const fn webgpu_only(&self) -> bool {
        !self.descriptor.weights.runs_on_cpu()
    }

    /// Whether every word of `query` appears in the name, id or tags, ignoring case.
    #[must_use]
    pub fn matches(&self, query: &str) -> bool {
        let haystack = format!(
            "{} {} {} {} {}",
            self.label,
            self.id,
            self.line().label(),
            languages_label(self.languages()),
            self.quantization_tag().unwrap_or_default()
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
    use crate::model_catalog::test_support::{info, parakeet_info, qwen3_info};
    use purr_common::model::family::{Qwen3AsrSize, WhisperSize};

    #[test]
    fn a_quantized_english_model_gets_a_friendly_label() {
        let entry = ModelEntry::from_info(&info("tiny.en-q5_1", "tiny", 1024, Some("q5_1"), true));
        assert_eq!(entry.name, "Tiny (English)");
        assert_eq!(entry.label, "Tiny (English) · Q5");
        assert_eq!(entry.group, ModelGroup::Tiny);
        assert!(entry.is_english_only());
        assert!(!entry.is_multilingual());
    }

    #[test]
    fn a_multilingual_model_says_so() {
        let entry = ModelEntry::from_info(&info("base", "base", 1024, None, false));
        assert!(entry.is_multilingual());
        assert_eq!(languages_label(entry.languages()), "Multilingual");
        assert_eq!(entry.family(), ModelFamily::Whisper(WhisperSize::Base));
    }

    #[test]
    fn english_only_models_are_tagged_english_only() {
        assert_eq!(
            languages_label(&LanguageSupport::english_only()),
            "English only"
        );
    }

    #[test]
    fn precision_comes_from_the_weights() {
        let tag = |weights| {
            let model = ModelEntry::from_info(&info("m", "tiny", 1, weights, false));
            precision_tag(model.precision())
        };
        assert_eq!(tag(None), None);
        assert_eq!(tag(Some("q5_k_m")), Some("Quantized Q5".to_owned()));
        assert_eq!(tag(Some("q6_k")), Some("Quantized Q6".to_owned()));
        assert_eq!(tag(Some("fp32/q4")), None);
        assert_eq!(tag(Some("q4/q4")), Some("Quantized Q4".to_owned()));
    }

    #[test]
    fn a_model_with_a_list_of_languages_says_how_many() {
        let entry = ModelEntry::from_info(&parakeet_info("parakeet-q8", "q8_0", 700, false));
        assert_eq!(languages_label(entry.languages()), "25 languages");
        assert!(entry.is_multilingual());
        assert!(!entry.is_english_only());
        assert_eq!(entry.line(), ModelLine::Parakeet);
        assert_eq!(entry.group, ModelGroup::Parakeet);
    }

    #[test]
    fn the_quantization_tag_names_the_stored_format() {
        let tag = |model: ModelInfo| ModelEntry::from_info(&model).quantization_tag();
        assert_eq!(
            tag(parakeet_info("p", "f16", 1, false)),
            Some("F16".to_owned())
        );
        assert_eq!(
            tag(parakeet_info("p", "q4_k_m", 1, false)),
            Some("Quantized Q4_K_M".to_owned())
        );
        assert_eq!(tag(info("base", "base", 1, Some("fp32/q4"), false)), None);
        assert_eq!(
            tag(info("base", "base", 1, Some("q4/q4"), false)),
            Some("Quantized Q4".to_owned())
        );
    }

    #[test]
    fn a_qwen3_asr_model_is_its_own_line_and_group_with_30_languages() {
        let entry = ModelEntry::from_info(&qwen3_info("q", Qwen3AsrSize::B1_7, "q8_0", 1));
        assert_eq!(languages_label(entry.languages()), "30 languages");
        assert_eq!(entry.line(), ModelLine::Qwen3Asr);
        assert_eq!(entry.group, ModelGroup::Qwen3Asr);
    }

    fn search_finds_a_model_by_its_line_and_its_quantization() {
        let entry = ModelEntry::from_info(&parakeet_info("p-q4", "q4_k_m", 1, false));
        assert!(entry.matches("parakeet"));
        assert!(entry.matches("q4_k_m 25"));
        assert!(!entry.matches("whisper"));
    }

    #[test]
    fn web_models_group_with_their_size_class() {
        let entry = ModelEntry::from_info(&info("base.en", "base", 1024, Some("fp32/q4"), true));
        assert_eq!(entry.family(), ModelFamily::Whisper(WhisperSize::Base));
        assert_eq!(entry.label, "Base (English)");
        assert_eq!(entry.precision(), Precision::Mixed);
    }

    #[test]
    fn search_needs_every_word_and_ignores_case() {
        let entry =
            ModelEntry::from_info(&info("small.en-q5_1", "small", 1024, Some("q5_1"), true));
        assert!(entry.matches("SMALL english"));
        assert!(entry.matches("q5"));
        assert!(entry.matches(""));
        assert!(!entry.matches("small medium"));
    }
}
