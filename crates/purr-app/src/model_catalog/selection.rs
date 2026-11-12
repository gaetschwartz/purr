use super::{
    listing::ModelRow,
    recommend::{Device, recommend},
};

/// The model new transcriptions use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    /// The saved model, found in the catalog.
    Saved(ModelRow),
    /// The saved id is not in this catalog, so a suggested model stands in for it.
    Substitute(ModelRow),
    /// Neither the saved model nor any suggestion exists here.
    Missing,
}

impl Selection {
    /// The model to use, if there is one.
    #[must_use]
    pub const fn row(&self) -> Option<&ModelRow> {
        match self {
            Self::Saved(row) | Self::Substitute(row) => Some(row),
            Self::Missing => None,
        }
    }
}

/// Finds `saved` in `rows`, or the model suggested for `device` when it is not there.
#[must_use]
pub fn resolve(rows: &[ModelRow], saved: &str, device: Device) -> Selection {
    if let Some(row) = rows.iter().find(|row| row.entry.id == saved) {
        return Selection::Saved(row.clone());
    }
    recommend(rows, device)
        .and_then(|entry| rows.iter().find(|row| row.entry.id == entry.id))
        .map_or(Selection::Missing, |row| Selection::Substitute(row.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_catalog::{entry::ModelEntry, state::ModelState};
    use purr_common::platform::{ModelInfo, ModelMetadata};

    fn row(id: &str, family: &str, size: u64) -> ModelRow {
        let metadata = ModelMetadata::new(
            "ggml".to_owned(),
            "whisper".to_owned(),
            "multilingual".to_owned(),
            "1".to_owned(),
        )
        .with_platform_attribute("family".to_owned(), family.to_owned());
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

    fn catalog() -> Vec<ModelRow> {
        vec![row("tiny", "tiny", 75), row("base", "base", 142)]
    }

    #[test]
    fn the_saved_model_wins_even_when_it_is_not_downloaded() {
        assert_eq!(
            resolve(&catalog(), "tiny", Device::Browser),
            Selection::Saved(row("tiny", "tiny", 75))
        );
    }

    #[test]
    fn an_unknown_saved_id_falls_back_to_the_suggestion() {
        assert_eq!(
            resolve(&catalog(), "gone", Device::Browser),
            Selection::Substitute(row("base", "base", 142))
        );
    }

    #[test]
    fn an_empty_catalog_has_no_selection() {
        assert_eq!(resolve(&[], "base", Device::Browser), Selection::Missing);
    }
}
