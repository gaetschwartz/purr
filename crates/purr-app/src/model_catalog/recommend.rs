use super::{
    entry::{Language, ModelEntry, Precision},
    family::ModelGroup,
    listing::ModelRow,
    state::ModelState,
};

/// What the device running Purr can do, for picking a model when none usable is saved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Desktop { gpu: bool },
    Browser,
}

impl Device {
    /// Size classes to suggest, best first.
    const fn groups(self) -> &'static [ModelGroup] {
        match self {
            Self::Desktop { gpu: true } => &[ModelGroup::Turbo, ModelGroup::Base, ModelGroup::Tiny],
            Self::Desktop { gpu: false } | Self::Browser => &[ModelGroup::Base, ModelGroup::Tiny],
        }
    }
}

impl Device {
    /// Whether a quantized model of the catalog is a fair suggestion here. Browser quantization
    /// only shrinks the download; on WebGPU it runs slower than the unquantized weights.
    const fn suggests_quantized(self) -> bool {
        match self {
            Self::Desktop { .. } => true,
            Self::Browser => false,
        }
    }
}

/// The smallest multilingual model of `group` that runs on `device`.
fn pick(rows: &[ModelRow], group: ModelGroup, device: Device) -> Option<&ModelEntry> {
    rows.iter()
        .filter(|row| {
            row.entry.group == group
                && row.entry.language == Language::Multilingual
                && !row.entry.speaker_turns
                && !matches!(row.state, ModelState::Unavailable { .. })
                && (device.suggests_quantized()
                    || !matches!(row.entry.precision, Precision::Quantized { .. }))
        })
        .map(|row| &row.entry)
        .min_by_key(|entry| entry.size_bytes)
}

/// The model to suggest from the real catalog in `rows` for `device`; `None` when nothing fits.
#[must_use]
pub fn recommend(rows: &[ModelRow], device: Device) -> Option<ModelEntry> {
    device
        .groups()
        .iter()
        .find_map(|group| pick(rows, *group, device))
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_catalog::state::UnavailableReason;
    use purr_common::platform::{ModelInfo, ModelMetadata, model_flag};

    fn row(id: &str, family: &str, size: u64, english: bool) -> ModelRow {
        row_with_quantization(id, family, size, english, None)
    }

    fn row_with_quantization(
        id: &str,
        family: &str,
        size: u64,
        english: bool,
        quantization: Option<&str>,
    ) -> ModelRow {
        let mut metadata = ModelMetadata::new(
            "ggml".to_owned(),
            "whisper".to_owned(),
            "multilingual".to_owned(),
            "1".to_owned(),
        )
        .with_platform_attribute("family".to_owned(), family.to_owned())
        .with_platform_attribute(model_flag::ENGLISH_ONLY.to_owned(), english.to_string());
        if let Some(quantization) = quantization {
            metadata = metadata.with_quantization(quantization.to_owned());
        }
        ModelRow {
            entry: ModelEntry::from_info(&ModelInfo::new(
                id.to_owned(),
                id.to_owned(),
                String::new(),
                size,
                false,
                metadata,
            )),
            state: ModelState::NotDownloaded,
        }
    }

    fn desktop_catalog() -> Vec<ModelRow> {
        vec![
            row("tiny", "tiny", 75, false),
            row("tiny-q5_1", "tiny", 31, false),
            row("base", "base", 142, false),
            row("base.en", "base", 142, true),
            row("base-q5_1", "base", 57, false),
            row("large-v3-turbo", "large-v3-turbo", 1_620, false),
            row("large-v3-turbo-q5_0", "large-v3-turbo", 547, false),
        ]
    }

    fn id(recommendation: Option<ModelEntry>) -> Option<String> {
        recommendation.map(|entry| entry.id)
    }

    #[test]
    fn a_desktop_with_a_gpu_gets_the_quantized_turbo() {
        assert_eq!(
            id(recommend(&desktop_catalog(), Device::Desktop { gpu: true })),
            Some("large-v3-turbo-q5_0".to_owned())
        );
    }

    #[test]
    fn a_desktop_without_a_gpu_gets_a_small_base() {
        assert_eq!(
            id(recommend(
                &desktop_catalog(),
                Device::Desktop { gpu: false }
            )),
            Some("base-q5_1".to_owned())
        );
    }

    #[test]
    fn the_browser_gets_base_and_skips_models_it_cannot_run() {
        let mut catalog = vec![
            row("tiny", "tiny", 117, false),
            row("base", "base", 199, false),
            row("base.en", "base.en", 199, true),
            row("large-v3-turbo", "large-v3-turbo", 900, false),
        ];
        assert_eq!(
            id(recommend(&catalog, Device::Browser)),
            Some("base".to_owned())
        );
        for entry in &mut catalog {
            if entry.entry.id == "base" {
                entry.state = ModelState::Unavailable {
                    reason: UnavailableReason::NeedsShaderF16,
                    installed: false,
                };
            }
        }
        assert_eq!(
            id(recommend(&catalog, Device::Browser)),
            Some("tiny".to_owned())
        );
    }

    #[test]
    fn the_browser_never_suggests_a_quantized_variant_over_the_unquantized_one() {
        let catalog = vec![
            row_with_quantization("tiny", "tiny", 122, false, Some("fp32/q4")),
            row_with_quantization("base", "base", 209, false, Some("fp32/q4")),
            row_with_quantization("base-q4", "base", 145, false, Some("q4/q4")),
        ];
        assert_eq!(
            id(recommend(&catalog, Device::Browser)),
            Some("base".to_owned())
        );
        assert_eq!(
            id(recommend(&catalog, Device::Desktop { gpu: false })),
            Some("base-q4".to_owned())
        );
    }

    #[test]
    fn an_empty_catalog_recommends_nothing() {
        assert_eq!(recommend(&[], Device::Browser), None);
    }
}
