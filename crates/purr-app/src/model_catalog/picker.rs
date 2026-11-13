use super::{
    clusters::{ModelCluster, cluster},
    listing::ModelRow,
    state::ModelState,
};

/// The model popup's list: what can run right away, then what would be downloaded first.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PickerSections {
    /// Every version stored on this device, smallest first.
    pub on_device: Vec<ModelRow>,
    /// Models not on this device, with their other versions folded in: busy ones first, then by size.
    pub download: Vec<ModelCluster>,
    /// Models a newer family replaces that are not on this device.
    pub older: Vec<ModelCluster>,
}

impl PickerSections {
    /// Whether nothing matches.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.on_device.is_empty() && self.download.is_empty() && self.older.is_empty()
    }

    /// Whether the older models show without being asked: a search reached them or one is selected.
    #[must_use]
    pub fn older_opens(&self, search: &str, selected: &str) -> bool {
        !self.older.is_empty()
            && (!search.trim().is_empty()
                || self.older.iter().any(|cluster| cluster.holds(selected)))
    }

    /// Every model the list offers, in the order it shows them.
    #[must_use]
    pub fn choices(&self, older_open: bool) -> Vec<&ModelRow> {
        let older: &[ModelCluster] = if older_open { &self.older } else { &[] };
        self.on_device
            .iter()
            .chain(
                self.download
                    .iter()
                    .chain(older)
                    .flat_map(ModelCluster::rows),
            )
            .collect()
    }
}

fn needs_attention(cluster: &ModelCluster) -> bool {
    cluster.rows().any(|row| match row.state {
        ModelState::Downloading(_)
        | ModelState::Verifying(_)
        | ModelState::Failed(_)
        | ModelState::NeedsRepair(_) => true,
        ModelState::NotDownloaded | ModelState::Downloaded | ModelState::Unavailable { .. } => {
            false
        }
    })
}

/// Whether the popup counts `row` as on this device: stored, or stored and being checked.
#[must_use]
pub fn stored_here(row: &ModelRow) -> bool {
    row.state.is_on_device()
        || (row.entry.installed && matches!(row.state, ModelState::Verifying(_)))
}

/// Sorts the models matching `search` into the popup's sections.
#[must_use]
pub fn picker_sections(rows: &[ModelRow], search: &str) -> PickerSections {
    let search = search.trim();
    let (mut on_device, mut elsewhere): (Vec<ModelRow>, Vec<ModelRow>) = rows
        .iter()
        .filter(|row| row.entry.matches(search))
        .cloned()
        .partition(stored_here);
    on_device.sort_by_key(|row| row.entry.size_bytes);
    elsewhere.sort_by_key(|row| row.entry.size_bytes);
    let (older, mut download): (Vec<ModelCluster>, Vec<ModelCluster>) = cluster(&elsewhere)
        .into_iter()
        .partition(ModelCluster::superseded);
    download.sort_by_key(|cluster| (!needs_attention(cluster), cluster.primary.entry.size_bytes));
    PickerSections {
        on_device,
        download,
        older,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_catalog::{
        activity::{DownloadProgress, VerifyProgress},
        entry::ModelEntry,
    };
    use purr_common::platform::{ModelInfo, ModelMetadata};

    fn row(id: &str, family: &str, size: u64, quantization: Option<&str>) -> ModelRow {
        let mut metadata = ModelMetadata::new(
            "ggml".to_owned(),
            "whisper".to_owned(),
            "multilingual".to_owned(),
            "v1".to_owned(),
        )
        .with_platform_attribute("family".to_owned(), family.to_owned());
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

    fn stored(mut row: ModelRow) -> ModelRow {
        row.state = ModelState::Downloaded;
        row
    }

    fn catalog() -> Vec<ModelRow> {
        vec![
            row("large-v3-turbo", "large-v3-turbo", 1_600, None),
            stored(row(
                "large-v3-turbo-q5",
                "large-v3-turbo",
                550,
                Some("q5_0"),
            )),
            row("small", "small", 480, None),
            stored(row("base", "base", 140, None)),
            row("tiny", "tiny", 75, None),
            row("tiny-q5", "tiny", 31, Some("q5_1")),
            row("large-v2", "large-v2", 3_000, None),
        ]
    }

    fn ids<'a>(rows: impl IntoIterator<Item = &'a ModelRow>) -> Vec<&'a str> {
        rows.into_iter().map(|row| row.entry.id.as_str()).collect()
    }

    fn shape(clusters: &[ModelCluster]) -> Vec<(&str, Vec<&str>)> {
        clusters
            .iter()
            .map(|cluster| (cluster.primary.entry.id.as_str(), ids(&cluster.variants)))
            .collect()
    }

    #[test]
    fn stored_versions_come_first_each_on_its_own_and_smallest_first() {
        let sections = picker_sections(&catalog(), "");
        assert_eq!(ids(&sections.on_device), ["base", "large-v3-turbo-q5"]);
    }

    #[test]
    fn models_to_download_fold_their_versions_and_go_by_size() {
        let sections = picker_sections(&catalog(), "");
        assert_eq!(
            shape(&sections.download),
            [
                ("tiny", vec!["tiny-q5"]),
                ("small", vec![]),
                ("large-v3-turbo", vec![])
            ]
        );
        assert_eq!(shape(&sections.older), [("large-v2", vec![])]);
    }

    #[test]
    fn a_busy_download_leads_the_download_section() {
        let mut rows = catalog();
        rows[2].state = ModelState::Downloading(DownloadProgress::default());
        let sections = picker_sections(&rows, "");
        assert_eq!(sections.download[0].primary.entry.id, "small");
    }

    #[test]
    fn a_stored_model_being_checked_stays_on_the_device() {
        let mut rows = catalog();
        rows[3].entry.installed = true;
        rows[3].state = ModelState::Verifying(VerifyProgress::default());
        let sections = picker_sections(&rows, "");
        assert_eq!(ids(&sections.on_device), ["base", "large-v3-turbo-q5"]);
    }

    #[test]
    fn a_selected_model_that_is_not_stored_is_still_listed() {
        let sections = picker_sections(&catalog(), "");
        assert!(
            sections
                .choices(false)
                .iter()
                .any(|row| row.entry.id == "small")
        );
    }

    #[test]
    fn the_older_models_open_for_a_search_or_when_one_is_selected() {
        let sections = picker_sections(&catalog(), "");
        assert!(!sections.older_opens("", "base"));
        assert!(sections.older_opens("", "large-v2"));
        let found = picker_sections(&catalog(), "large");
        assert_eq!(shape(&found.older), [("large-v2", vec![])]);
        assert!(found.older_opens("large", "base"));
    }

    #[test]
    fn choices_follow_the_order_on_screen() {
        let sections = picker_sections(&catalog(), "");
        assert_eq!(
            ids(sections.choices(false)),
            [
                "base",
                "large-v3-turbo-q5",
                "tiny",
                "tiny-q5",
                "small",
                "large-v3-turbo"
            ]
        );
        assert_eq!(ids(sections.choices(true)).last(), Some(&"large-v2"));
    }

    #[test]
    fn a_search_can_leave_nothing() {
        assert!(picker_sections(&catalog(), "zzz").is_empty());
        assert!(!picker_sections(&catalog(), "  tiny ").is_empty());
    }
}
