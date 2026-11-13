use super::{
    clusters::{ModelCluster, cluster},
    entry::{Language, ModelEntry},
    family::ModelGroup,
    state::ModelState,
};

/// A model with where it stands right now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRow {
    pub entry: ModelEntry,
    pub state: ModelState,
}

/// Which models the filter control keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelFilter {
    #[default]
    All,
    OnDevice,
    Multilingual,
    EnglishOnly,
}

impl ModelFilter {
    /// The filters in the order the control shows them.
    pub const ALL: [Self; 4] = [
        Self::All,
        Self::OnDevice,
        Self::Multilingual,
        Self::EnglishOnly,
    ];

    /// The segment text.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::OnDevice => "On this device",
            Self::Multilingual => "Multilingual",
            Self::EnglishOnly => "English only",
        }
    }

    fn keeps(self, row: &ModelRow) -> bool {
        match self {
            Self::All => true,
            Self::OnDevice => row.state.is_on_device(),
            Self::Multilingual => row.entry.language == Language::Multilingual,
            Self::EnglishOnly => row.entry.language == Language::EnglishOnly,
        }
    }
}

/// What the user asked to see.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ModelQuery {
    pub filter: ModelFilter,
    pub group: Option<ModelGroup>,
    pub search: String,
}

impl ModelQuery {
    fn is_unfiltered(&self) -> bool {
        self.filter == ModelFilter::All && self.group.is_none() && self.search.trim().is_empty()
    }

    fn keeps(&self, row: &ModelRow) -> bool {
        self.filter.keeps(row)
            && self.group.is_none_or(|group| group == row.entry.group)
            && row.entry.matches(self.search.trim())
    }
}

/// The models of one list section, each with its quantized versions folded in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupListing {
    pub group: ModelGroup,
    pub clusters: Vec<ModelCluster>,
}

/// The list the Models screen shows.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Listing {
    /// Downloads, checks and failures, shown first when nothing is filtered.
    pub in_progress: Vec<ModelCluster>,
    pub groups: Vec<GroupListing>,
    /// Models a newer family replaces, kept out of the sections.
    pub more: Vec<ModelCluster>,
}

impl Listing {
    /// Whether nothing matches.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.in_progress.is_empty() && self.groups.is_empty() && self.more.is_empty()
    }

    /// Whether the older models should be shown without being asked: a search reached them, or one
    /// of them is stored, busy or in use.
    #[must_use]
    pub fn shows_more(&self, query: &ModelQuery, in_use: Option<&str>) -> bool {
        !self.more.is_empty()
            && (!query.search.trim().is_empty()
                || self.more.iter().any(|cluster| {
                    in_use.is_some_and(|id| cluster.holds(id))
                        || cluster
                            .rows()
                            .any(|row| row.state.is_on_device() || row.state.is_active())
                }))
    }
}

fn is_busy(cluster: &ModelCluster) -> bool {
    cluster.rows().any(|row| row.state.is_active())
}

/// Filters `rows`, folds quantized versions into their model and sorts the result into sections,
/// keeping the platform's order inside each.
#[must_use]
pub fn build_listing(rows: &[ModelRow], query: &ModelQuery) -> Listing {
    let pin_active = query.is_unfiltered();
    let kept: Vec<ModelRow> = rows
        .iter()
        .filter(|row| query.keeps(row))
        .cloned()
        .collect();
    let mut listing = Listing::default();
    for cluster in cluster(&kept) {
        if pin_active && is_busy(&cluster) {
            listing.in_progress.push(cluster);
        } else if cluster.superseded() {
            listing.more.push(cluster);
        } else {
            let group = cluster.primary.entry.group;
            match listing
                .groups
                .iter_mut()
                .find(|section| section.group == group)
            {
                Some(section) => section.clusters.push(cluster),
                None => listing.groups.push(GroupListing {
                    group,
                    clusters: vec![cluster],
                }),
            }
        }
    }
    listing.groups.sort_by_key(|section| {
        ModelGroup::ORDER
            .iter()
            .position(|group| *group == section.group)
    });
    listing
}

/// How many models are stored and the space they take.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    pub installed: usize,
    pub total: usize,
    pub used_bytes: u64,
}

/// Counts the models stored on this device and the space they take.
#[must_use]
pub fn summarize(rows: &[ModelRow]) -> Summary {
    let stored = rows.iter().filter(|row| row.entry.installed);
    Summary {
        installed: stored.clone().count(),
        total: rows.len(),
        used_bytes: stored.map(|row| row.entry.size_bytes).sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_catalog::activity::{DownloadFailure, DownloadProgress};
    use purr_common::platform::{ModelFailureKind, ModelInfo, ModelMetadata, model_flag};

    fn row(
        id: &str,
        family: &str,
        english: bool,
        quantization: Option<&str>,
        state: ModelState,
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
                100,
                state.is_on_device(),
                metadata,
            )),
            state,
        }
    }

    fn plain(id: &str, family: &str, state: ModelState) -> ModelRow {
        row(id, family, false, None, state)
    }

    fn failed() -> ModelState {
        ModelState::Failed(DownloadFailure {
            kind: ModelFailureKind::Other,
            message: String::new(),
        })
    }

    fn rows() -> Vec<ModelRow> {
        vec![
            plain("tiny", "tiny", ModelState::Downloaded),
            row(
                "tiny-q5",
                "tiny",
                false,
                Some("q5_1"),
                ModelState::NotDownloaded,
            ),
            row("tiny.en", "tiny", true, None, ModelState::NotDownloaded),
            plain(
                "small",
                "small",
                ModelState::Downloading(DownloadProgress::default()),
            ),
            plain("base", "base", failed()),
            plain("large-v3", "large-v3", ModelState::NotDownloaded),
            plain("large-v2", "large-v2", ModelState::NotDownloaded),
            plain("large-v1", "large-v1", ModelState::Downloaded),
            plain("mystery", "mystery", ModelState::NotDownloaded),
        ]
    }

    fn shape(clusters: &[ModelCluster]) -> Vec<(&str, Vec<&str>)> {
        clusters
            .iter()
            .map(|cluster| {
                (
                    cluster.primary.entry.id.as_str(),
                    cluster
                        .variants
                        .iter()
                        .map(|row| row.entry.id.as_str())
                        .collect(),
                )
            })
            .collect()
    }

    fn section(listing: &Listing, group: ModelGroup) -> Vec<(&str, Vec<&str>)> {
        listing
            .groups
            .iter()
            .find(|section| section.group == group)
            .map(|section| shape(&section.clusters))
            .unwrap_or_default()
    }

    fn query(filter: ModelFilter) -> ModelQuery {
        ModelQuery {
            filter,
            ..ModelQuery::default()
        }
    }

    fn search(text: &str) -> ModelQuery {
        ModelQuery {
            search: text.to_owned(),
            ..ModelQuery::default()
        }
    }

    #[test]
    fn sections_follow_the_prototype_order_and_other_comes_last() {
        let listing = build_listing(&rows(), &ModelQuery::default());
        let order: Vec<ModelGroup> = listing.groups.iter().map(|section| section.group).collect();
        assert_eq!(
            order,
            [ModelGroup::Large, ModelGroup::Tiny, ModelGroup::Other]
        );
    }

    #[test]
    fn quantized_versions_fold_under_their_model() {
        let listing = build_listing(&rows(), &ModelQuery::default());
        assert_eq!(
            section(&listing, ModelGroup::Tiny),
            [("tiny", vec!["tiny-q5"]), ("tiny.en", vec![])]
        );
    }

    #[test]
    fn active_models_move_to_the_top_when_nothing_is_filtered() {
        let listing = build_listing(&rows(), &ModelQuery::default());
        assert_eq!(
            shape(&listing.in_progress),
            [("small", vec![]), ("base", vec![])]
        );
    }

    #[test]
    fn a_busy_variant_pulls_its_whole_model_to_the_top() {
        let mut rows = rows();
        rows[1].state = ModelState::Downloading(DownloadProgress::default());
        let listing = build_listing(&rows, &ModelQuery::default());
        assert_eq!(shape(&listing.in_progress)[0], ("tiny", vec!["tiny-q5"]));
        assert_eq!(section(&listing, ModelGroup::Tiny), [("tiny.en", vec![])]);
    }

    #[test]
    fn superseded_models_leave_the_sections_for_more_models() {
        let listing = build_listing(&rows(), &ModelQuery::default());
        assert_eq!(section(&listing, ModelGroup::Large), [("large-v3", vec![])]);
        assert_eq!(
            shape(&listing.more),
            [("large-v2", vec![]), ("large-v1", vec![])]
        );
    }

    #[test]
    fn a_filter_keeps_active_models_in_their_sections() {
        let listing = build_listing(&rows(), &query(ModelFilter::Multilingual));
        assert_eq!(listing.in_progress, []);
        let order: Vec<ModelGroup> = listing.groups.iter().map(|section| section.group).collect();
        assert_eq!(
            order,
            [
                ModelGroup::Large,
                ModelGroup::Small,
                ModelGroup::Base,
                ModelGroup::Tiny,
                ModelGroup::Other
            ]
        );
    }

    #[test]
    fn on_device_keeps_only_stored_models() {
        let listing = build_listing(&rows(), &query(ModelFilter::OnDevice));
        assert_eq!(listing.groups.len(), 1);
        assert_eq!(section(&listing, ModelGroup::Tiny), [("tiny", vec![])]);
        assert_eq!(shape(&listing.more), [("large-v1", vec![])]);
    }

    #[test]
    fn on_device_still_shows_a_model_whose_only_stored_version_is_quantized() {
        let mut rows = rows();
        rows[0].state = ModelState::NotDownloaded;
        rows[1].state = ModelState::Downloaded;
        let listing = build_listing(&rows, &query(ModelFilter::OnDevice));
        assert_eq!(section(&listing, ModelGroup::Tiny), [("tiny-q5", vec![])]);
    }

    #[test]
    fn on_device_keeps_stored_variants_with_their_stored_model() {
        let mut rows = rows();
        rows[1].state = ModelState::Downloaded;
        let listing = build_listing(&rows, &query(ModelFilter::OnDevice));
        assert_eq!(
            section(&listing, ModelGroup::Tiny),
            [("tiny", vec!["tiny-q5"])]
        );
    }

    #[test]
    fn english_only_and_group_filters_combine() {
        let query = ModelQuery {
            filter: ModelFilter::EnglishOnly,
            group: Some(ModelGroup::Tiny),
            search: String::new(),
        };
        let listing = build_listing(&rows(), &query);
        assert_eq!(listing.groups.len(), 1);
        assert_eq!(section(&listing, ModelGroup::Tiny), [("tiny.en", vec![])]);
    }

    #[test]
    fn search_reaches_the_older_models_and_a_lone_variant() {
        let listing = build_listing(&rows(), &search("  large "));
        assert_eq!(section(&listing, ModelGroup::Large), [("large-v3", vec![])]);
        assert_eq!(
            shape(&listing.more),
            [("large-v2", vec![]), ("large-v1", vec![])]
        );
        let variant = build_listing(&rows(), &search("q5"));
        assert_eq!(section(&variant, ModelGroup::Tiny), [("tiny-q5", vec![])]);
    }

    #[test]
    fn search_can_leave_nothing() {
        assert!(build_listing(&rows(), &search("zzz")).is_empty());
    }

    #[test]
    fn the_older_models_open_for_a_search_a_stored_model_or_the_one_in_use() {
        let listing = build_listing(&rows(), &ModelQuery::default());
        assert!(listing.shows_more(&ModelQuery::default(), None));

        let mut none_stored = rows();
        none_stored[7].state = ModelState::NotDownloaded;
        let listing = build_listing(&none_stored, &ModelQuery::default());
        assert!(!listing.shows_more(&ModelQuery::default(), None));
        assert!(listing.shows_more(&ModelQuery::default(), Some("large-v2")));
        assert!(!listing.shows_more(&ModelQuery::default(), Some("tiny")));

        let found = build_listing(&none_stored, &search("large"));
        assert!(found.shows_more(&search("large"), None));
    }

    #[test]
    fn the_older_models_stay_closed_when_there_are_none() {
        let listing = build_listing(&rows(), &search("tiny"));
        assert!(!listing.shows_more(&search("tiny"), None));
    }

    #[test]
    fn summary_counts_stored_models_and_their_space() {
        let summary = summarize(&rows());
        assert_eq!(summary.installed, 2);
        assert_eq!(summary.total, 9);
        assert_eq!(summary.used_bytes, 200);
    }
}
