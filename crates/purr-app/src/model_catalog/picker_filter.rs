use super::{entry::Precision, listing::ModelRow};

/// Which stored state the model popup lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PickerStorage {
    #[default]
    OnDevice,
    All,
}

/// Which weight precisions the model popup lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PickerPrecision {
    #[default]
    All,
    Full,
    Quantized,
}

/// What the model popup was asked to show.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PickerQuery {
    pub storage: PickerStorage,
    pub precision: PickerPrecision,
    pub search: String,
}

impl PickerQuery {
    fn keeps(&self, row: &ModelRow) -> bool {
        let storage = match self.storage {
            PickerStorage::OnDevice => row.state.is_on_device(),
            PickerStorage::All => true,
        };
        let quantized = matches!(row.entry.precision, Precision::Quantized { .. });
        let precision = match self.precision {
            PickerPrecision::All => true,
            PickerPrecision::Full => !quantized,
            PickerPrecision::Quantized => quantized,
        };
        storage && precision && row.entry.matches(self.search.trim())
    }
}

/// The storage filter a popup opens with: what is on this device, or everything when nothing is.
#[must_use]
pub fn opening_storage(rows: &[ModelRow]) -> PickerStorage {
    if rows.iter().any(|row| row.state.is_on_device()) {
        PickerStorage::OnDevice
    } else {
        PickerStorage::All
    }
}

/// The rows `query` keeps, models on this device first and then by size.
#[must_use]
pub fn pick_rows(rows: &[ModelRow], query: &PickerQuery) -> Vec<ModelRow> {
    let mut kept: Vec<ModelRow> = rows
        .iter()
        .filter(|row| query.keeps(row))
        .cloned()
        .collect();
    kept.sort_by_key(|row| (!row.state.is_on_device(), row.entry.size_bytes));
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_catalog::{entry::ModelEntry, state::ModelState};
    use purr_common::platform::{ModelInfo, ModelMetadata};

    fn row(id: &str, size: u64, quantization: Option<&str>, state: ModelState) -> ModelRow {
        let mut metadata = ModelMetadata::new(
            "ggml".to_owned(),
            "whisper".to_owned(),
            "multilingual".to_owned(),
            "v1".to_owned(),
        );
        if let Some(quantization) = quantization {
            metadata = metadata.with_quantization(quantization.to_owned());
        }
        ModelRow {
            entry: ModelEntry::from_info(&ModelInfo::new(
                id.to_owned(),
                id.to_owned(),
                String::new(),
                size,
                state == ModelState::Downloaded,
                metadata,
            )),
            state,
        }
    }

    fn rows() -> Vec<ModelRow> {
        vec![
            row("tiny", 10, None, ModelState::NotDownloaded),
            row("base-q5", 20, Some("q5_1"), ModelState::Downloaded),
            row("base", 30, None, ModelState::Downloaded),
            row("small-q5", 40, Some("q5_1"), ModelState::NotDownloaded),
        ]
    }

    fn ids(rows: &[ModelRow]) -> Vec<&str> {
        rows.iter().map(|row| row.entry.id.as_str()).collect()
    }

    #[test]
    fn the_popup_opens_on_the_device_when_anything_is_there() {
        assert_eq!(opening_storage(&rows()), PickerStorage::OnDevice);
        assert_eq!(opening_storage(&rows()[..1]), PickerStorage::All);
    }

    #[test]
    fn models_on_the_device_come_first_then_by_size() {
        let query = PickerQuery {
            storage: PickerStorage::All,
            ..PickerQuery::default()
        };
        assert_eq!(
            ids(&pick_rows(&rows(), &query)),
            ["base-q5", "base", "tiny", "small-q5"]
        );
    }

    #[test]
    fn the_device_filter_hides_what_is_not_downloaded() {
        assert_eq!(
            ids(&pick_rows(&rows(), &PickerQuery::default())),
            ["base-q5", "base"]
        );
    }

    #[test]
    fn the_precision_filter_splits_quantized_from_full() {
        let all = |precision| PickerQuery {
            storage: PickerStorage::All,
            precision,
            ..PickerQuery::default()
        };
        assert_eq!(
            ids(&pick_rows(&rows(), &all(PickerPrecision::Quantized))),
            ["base-q5", "small-q5"]
        );
        assert_eq!(
            ids(&pick_rows(&rows(), &all(PickerPrecision::Full))),
            ["base", "tiny"]
        );
    }
}
