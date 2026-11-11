use super::{
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

/// The rows of one list section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupListing {
    pub group: ModelGroup,
    pub rows: Vec<ModelRow>,
}

/// The list the Models screen shows.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Listing {
    /// Downloads, checks and failures, shown first when nothing is filtered.
    pub in_progress: Vec<ModelRow>,
    pub groups: Vec<GroupListing>,
}

impl Listing {
    /// Whether nothing matches.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.in_progress.is_empty() && self.groups.iter().all(|group| group.rows.is_empty())
    }
}

/// Filters `rows` and sorts them into sections, keeping the platform's order inside each.
#[must_use]
pub fn build_listing(rows: Vec<ModelRow>, query: &ModelQuery) -> Listing {
    let pin_active = query.is_unfiltered();
    let mut listing = Listing::default();
    for row in rows.into_iter().filter(|row| query.keeps(row)) {
        if pin_active && row.state.is_active() {
            listing.in_progress.push(row);
            continue;
        }
        match listing
            .groups
            .iter_mut()
            .find(|section| section.group == row.entry.group)
        {
            Some(section) => section.rows.push(row),
            None => listing.groups.push(GroupListing {
                group: row.entry.group,
                rows: vec![row],
            }),
        }
    }
    listing.groups.sort_by_key(|section| {
        ModelGroup::ORDER
            .iter()
            .position(|group| *group == section.group)
    });
    listing
}

/// The numbers of the stat cards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary {
    pub installed: usize,
    pub total: usize,
    pub used_bytes: u64,
    pub catalog_bytes: u64,
}

impl Summary {
    /// Share of the whole catalog that is stored, between 0 and 1.
    #[must_use]
    pub fn used_fraction(&self) -> f64 {
        if self.catalog_bytes == 0 {
            0.0
        } else {
            (self.used_bytes as f64 / self.catalog_bytes as f64).clamp(0.0, 1.0)
        }
    }
}

/// Counts the models stored on this device and the space they take.
#[must_use]
pub fn summarize(rows: &[ModelRow]) -> Summary {
    let stored = rows.iter().filter(|row| row.entry.installed);
    Summary {
        installed: stored.clone().count(),
        total: rows.len(),
        used_bytes: stored.map(|row| row.entry.size_bytes).sum(),
        catalog_bytes: rows.iter().map(|row| row.entry.size_bytes).sum(),
    }
}

/// The download that is running, if any: its row and its progress.
#[must_use]
pub fn active_download(rows: &[ModelRow]) -> Option<&ModelRow> {
    rows.iter()
        .find(|row| matches!(row.state, ModelState::Downloading(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_catalog::activity::{DownloadFailure, DownloadProgress};
    use purr_common::platform::{ModelFailureKind, ModelInfo, ModelMetadata, model_flag};

    fn row(id: &str, family: &str, english: bool, installed: bool, state: ModelState) -> ModelRow {
        let metadata = ModelMetadata::new(
            "ggml".to_owned(),
            "whisper".to_owned(),
            "multilingual".to_owned(),
            "v1".to_owned(),
        )
        .with_platform_attribute("family".to_owned(), family.to_owned())
        .with_platform_attribute(model_flag::ENGLISH_ONLY.to_owned(), english.to_string());
        ModelRow {
            entry: ModelEntry::from_info(&ModelInfo::new(
                id.to_owned(),
                id.to_owned(),
                String::new(),
                100,
                installed,
                metadata,
            )),
            state,
        }
    }

    fn rows() -> Vec<ModelRow> {
        vec![
            row("tiny", "tiny", false, true, ModelState::Downloaded),
            row("tiny.en", "tiny", true, false, ModelState::NotDownloaded),
            row(
                "small",
                "small",
                false,
                false,
                ModelState::Downloading(DownloadProgress::default()),
            ),
            row(
                "base",
                "base",
                false,
                false,
                ModelState::Failed(DownloadFailure {
                    kind: ModelFailureKind::Other,
                    message: String::new(),
                }),
            ),
            row(
                "large-v3",
                "large-v3",
                false,
                false,
                ModelState::NotDownloaded,
            ),
            row(
                "mystery",
                "mystery",
                false,
                false,
                ModelState::NotDownloaded,
            ),
        ]
    }

    fn ids(rows: &[ModelRow]) -> Vec<&str> {
        rows.iter().map(|row| row.entry.id.as_str()).collect()
    }

    #[test]
    fn sections_follow_the_prototype_order_and_other_comes_last() {
        let listing = build_listing(rows(), &ModelQuery::default());
        let order: Vec<ModelGroup> = listing.groups.iter().map(|section| section.group).collect();
        assert_eq!(
            order,
            [ModelGroup::Large, ModelGroup::Tiny, ModelGroup::Other]
        );
    }

    #[test]
    fn active_models_move_to_the_top_when_nothing_is_filtered() {
        let listing = build_listing(rows(), &ModelQuery::default());
        assert_eq!(ids(&listing.in_progress), ["small", "base"]);
        let tiny = listing.groups.iter().find(|s| s.group == ModelGroup::Tiny);
        assert_eq!(tiny.map(|s| ids(&s.rows)), Some(vec!["tiny", "tiny.en"]));
    }

    #[test]
    fn a_filter_keeps_active_models_in_their_sections() {
        let query = ModelQuery {
            filter: ModelFilter::Multilingual,
            ..ModelQuery::default()
        };
        let listing = build_listing(rows(), &query);
        assert_eq!(listing.in_progress, []);
        assert_eq!(listing.groups.len(), 5);
    }

    #[test]
    fn on_device_keeps_only_stored_models() {
        let query = ModelQuery {
            filter: ModelFilter::OnDevice,
            ..ModelQuery::default()
        };
        let listing = build_listing(rows(), &query);
        assert_eq!(listing.groups.len(), 1);
        assert_eq!(
            listing.groups.first().map(|s| ids(&s.rows)),
            Some(vec!["tiny"])
        );
    }

    #[test]
    fn english_only_and_group_filters_combine() {
        let query = ModelQuery {
            filter: ModelFilter::EnglishOnly,
            group: Some(ModelGroup::Tiny),
            search: String::new(),
        };
        let listing = build_listing(rows(), &query);
        assert_eq!(listing.groups.len(), 1);
        assert_eq!(
            listing.groups.first().map(|s| ids(&s.rows)),
            Some(vec!["tiny.en"])
        );
    }

    #[test]
    fn search_narrows_and_can_leave_nothing() {
        let query = ModelQuery {
            search: "  large ".to_owned(),
            ..ModelQuery::default()
        };
        assert_eq!(build_listing(rows(), &query).groups.len(), 1);
        let nothing = ModelQuery {
            search: "zzz".to_owned(),
            ..ModelQuery::default()
        };
        assert!(build_listing(rows(), &nothing).is_empty());
    }

    #[test]
    fn summary_counts_stored_models_and_their_space() {
        let summary = summarize(&rows());
        assert_eq!(summary.installed, 1);
        assert_eq!(summary.total, 6);
        assert_eq!(summary.used_bytes, 100);
        assert_eq!(summary.catalog_bytes, 600);
        assert!((summary.used_fraction() - 1.0 / 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn the_active_download_is_found() {
        let rows = rows();
        assert_eq!(
            active_download(&rows).map(|row| row.entry.id.as_str()),
            Some("small")
        );
    }
}
