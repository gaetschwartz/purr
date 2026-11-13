use super::{
    entry::{Language, Precision},
    listing::ModelRow,
    state::ModelState,
};
use crate::utils::counted::counted;

/// A model together with its quantized versions, which share its name and language.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCluster {
    /// The full-precision model, or the first version there is when none is full.
    pub primary: ModelRow,
    /// The other versions, smallest first.
    pub variants: Vec<ModelRow>,
}

impl ModelCluster {
    /// Every row of the cluster, the primary first.
    pub fn rows(&self) -> impl Iterator<Item = &ModelRow> {
        std::iter::once(&self.primary).chain(&self.variants)
    }

    /// Whether any row of the cluster is stored on this device.
    #[must_use]
    pub fn on_device(&self) -> bool {
        self.rows().any(|row| row.state.is_on_device())
    }

    /// Whether a download, a check, a failure or a damaged install sits among the variants.
    #[must_use]
    pub fn variants_need_attention(&self) -> bool {
        self.variants
            .iter()
            .any(|row| row.state.is_active() || matches!(row.state, ModelState::NeedsRepair(_)))
    }

    /// Whether the row of model `id` belongs to the cluster.
    #[must_use]
    pub fn holds(&self, id: &str) -> bool {
        self.rows().any(|row| row.entry.id == id)
    }

    /// What the fold under the primary says, such as "2 quantized versions".
    #[must_use]
    pub fn variants_label(&self) -> String {
        if self.variants.iter().all(is_quantized) {
            counted(
                self.variants.len(),
                "quantized version",
                "quantized versions",
            )
        } else {
            counted(self.variants.len(), "more version", "more versions")
        }
    }

    /// Whether the models of the cluster are superseded by a newer family.
    #[must_use]
    pub fn superseded(&self) -> bool {
        self.primary
            .entry
            .family
            .is_some_and(|family| family.superseded())
    }
}

type ClusterKey = (String, Language, bool);

fn key(row: &ModelRow) -> ClusterKey {
    (
        row.entry.name.clone(),
        row.entry.language,
        row.entry.speaker_turns,
    )
}

fn is_quantized(row: &ModelRow) -> bool {
    matches!(row.entry.precision, Precision::Quantized { .. })
}

/// Folds quantized models into the model they were made from, keeping the order of `rows`.
#[must_use]
pub fn cluster(rows: &[ModelRow]) -> Vec<ModelCluster> {
    let mut clusters: Vec<(ClusterKey, Vec<ModelRow>)> = Vec::new();
    for row in rows {
        let row_key = key(row);
        match clusters.iter_mut().find(|(found, _)| *found == row_key) {
            Some((_, members)) => members.push(row.clone()),
            None => clusters.push((row_key, vec![row.clone()])),
        }
    }
    clusters
        .into_iter()
        .filter_map(|(_, mut members)| {
            let primary_at = members
                .iter()
                .position(|row| !is_quantized(row))
                .unwrap_or(0);
            if members.is_empty() {
                return None;
            }
            let primary = members.remove(primary_at);
            members.sort_by_key(|row| row.entry.size_bytes);
            Some(ModelCluster {
                primary,
                variants: members,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_catalog::{
        activity::{DownloadFailure, DownloadProgress},
        entry::ModelEntry,
        family::ModelFamily,
        state::ModelState,
    };
    use purr_common::platform::{ModelFailureKind, ModelInfo, ModelMetadata, model_flag};

    fn row(
        id: &str,
        family: &str,
        size: u64,
        quantization: Option<&str>,
        english: bool,
    ) -> ModelRow {
        let mut metadata = ModelMetadata::new(
            "ggml".to_owned(),
            "whisper".to_owned(),
            "multilingual".to_owned(),
            "v1".to_owned(),
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

    fn shape(clusters: &[ModelCluster]) -> Vec<(String, Vec<String>)> {
        clusters
            .iter()
            .map(|cluster| {
                (
                    cluster.primary.entry.id.clone(),
                    cluster
                        .variants
                        .iter()
                        .map(|row| row.entry.id.clone())
                        .collect(),
                )
            })
            .collect()
    }

    #[test]
    fn quantized_models_fold_into_the_full_one() {
        let rows = [
            row("base-q8", "base", 80, Some("q8_0"), false),
            row("base", "base", 140, None, false),
            row("base-q5", "base", 60, Some("q5_1"), false),
            row("base.en", "base", 140, None, true),
        ];
        assert_eq!(
            shape(&cluster(&rows)),
            vec![
                (
                    "base".to_owned(),
                    vec!["base-q5".to_owned(), "base-q8".to_owned()]
                ),
                ("base.en".to_owned(), vec![]),
            ]
        );
    }

    #[test]
    fn a_model_with_only_quantized_versions_leads_with_the_first() {
        let rows = [
            row("small-q5", "small", 180, Some("q5_1"), false),
            row("small-q8", "small", 250, Some("q8_0"), false),
        ];
        assert_eq!(
            shape(&cluster(&rows)),
            vec![("small-q5".to_owned(), vec!["small-q8".to_owned()])]
        );
    }

    #[test]
    fn the_fold_is_named_after_what_it_holds() {
        let rows = [
            row("base", "base", 140, None, false),
            row("base-q5", "base", 60, Some("q5_1"), false),
        ];
        assert_eq!(cluster(&rows)[0].variants_label(), "1 quantized version");
        let rows = [
            row("base", "base", 140, None, false),
            row("base-q5", "base", 60, Some("q5_1"), false),
            row("base-q8", "base", 80, Some("q8_0"), false),
        ];
        assert_eq!(cluster(&rows)[0].variants_label(), "2 quantized versions");
        let rows = [
            row("base", "base", 140, None, false),
            row("base-alt", "base", 150, None, false),
        ];
        assert_eq!(cluster(&rows)[0].variants_label(), "1 more version");
    }

    #[test]
    fn a_busy_or_damaged_variant_needs_attention_but_the_primary_does_not() {
        let mut rows = [
            row("base", "base", 140, None, false),
            row("base-q5", "base", 60, Some("q5_1"), false),
        ];
        assert!(!cluster(&rows)[0].variants_need_attention());
        rows[0].state = ModelState::Failed(DownloadFailure {
            kind: ModelFailureKind::Other,
            message: String::new(),
        });
        assert!(!cluster(&rows)[0].variants_need_attention());
        rows[1].state = ModelState::Downloading(DownloadProgress::default());
        assert!(cluster(&rows)[0].variants_need_attention());
    }

    #[test]
    fn a_cluster_holds_the_ids_of_all_its_rows() {
        let rows = [
            row("base", "base", 140, None, false),
            row("base-q5", "base", 60, Some("q5_1"), false),
        ];
        let clusters = cluster(&rows);
        assert_eq!(
            (clusters[0].holds("base-q5"), clusters[0].holds("tiny")),
            (true, false)
        );
    }

    #[test]
    fn only_the_old_large_families_are_superseded() {
        let old = cluster(&[row("large-v1", "large-v1", 1, None, false)]);
        let new = cluster(&[row("large-v3", "large-v3", 1, None, false)]);
        assert_eq!((old[0].superseded(), new[0].superseded()), (true, false));
        assert_eq!(ModelFamily::LargeV2.superseded(), true);
    }
}
