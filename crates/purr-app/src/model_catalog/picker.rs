use super::{
    clusters::cluster,
    copy,
    entry::Precision,
    family::ModelGroup,
    format::percent,
    listing::ModelRow,
    state::{ModelState, UnavailableReason},
};
use crate::utils::byte_size::ByteSize;

/// Which models the model picker lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickerScope {
    /// Models with a version on this device, the selected one, and any being downloaded or checked.
    Downloaded,
    All,
}

impl PickerScope {
    /// The scopes in the order the control shows them.
    pub const ALL: [Self; 2] = [Self::Downloaded, Self::All];

    /// The segment text.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Downloaded => "Downloaded",
            Self::All => "All",
        }
    }

    /// The scope the picker opens on: Downloaded once anything is stored here, All before.
    #[must_use]
    pub fn initial(rows: &[ModelRow]) -> Self {
        if rows.iter().any(stored_here) {
            Self::Downloaded
        } else {
            Self::All
        }
    }
}

/// What the picker shows about a model or one of its versions at a glance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    Stored,
    Download { bytes: u64 },
    Downloading { percent: Option<u32> },
    Checking,
    Failed,
    Damaged,
    Unavailable(UnavailableReason),
}

impl Mark {
    /// The mark of one version.
    #[must_use]
    pub fn of(row: &ModelRow) -> Self {
        match &row.state {
            ModelState::Downloaded => Self::Stored,
            ModelState::NotDownloaded => Self::Download {
                bytes: row.entry.size_bytes,
            },
            ModelState::Downloading(progress) => Self::Downloading {
                percent: percent(progress),
            },
            ModelState::Verifying(_) => Self::Checking,
            ModelState::Failed(_) => Self::Failed,
            ModelState::NeedsRepair(_) => Self::Damaged,
            ModelState::Unavailable { reason, .. } => Self::Unavailable(*reason),
        }
    }

    /// What the mark says in words, for tooltips and screen readers.
    #[must_use]
    pub fn text(self) -> String {
        match self {
            Self::Stored => "On this device".to_owned(),
            Self::Download { bytes } => format!("{} download", ByteSize(bytes)),
            Self::Downloading { percent } => percent.map_or_else(
                || "Downloading…".to_owned(),
                |done| format!("Downloading, {done}%"),
            ),
            Self::Checking => "Checking files…".to_owned(),
            Self::Failed => "Last download failed".to_owned(),
            Self::Damaged => "Files damaged, download again".to_owned(),
            Self::Unavailable(reason) => copy::unavailable(reason).1.to_owned(),
        }
    }

    /// Which mark a model shows when its versions differ: lower wins.
    const fn rank(self) -> u8 {
        match self {
            Self::Downloading { .. } => 0,
            Self::Checking => 1,
            Self::Failed => 2,
            Self::Damaged => 3,
            Self::Stored => 4,
            Self::Download { .. } => 5,
            Self::Unavailable(_) => 6,
        }
    }
}

/// The short name of a version inside its model: `Full`, `Q8`, `Q5`.
#[must_use]
pub fn version_tag(precision: &Precision) -> String {
    match precision {
        Precision::Full | Precision::Mixed => "Full".to_owned(),
        Precision::Quantized { bits } => format!("Q{bits}"),
        Precision::Other(raw) => raw.clone(),
    }
}

/// Whether the picker counts `row` as on this device: stored, or stored and being checked.
#[must_use]
pub fn stored_here(row: &ModelRow) -> bool {
    row.state.is_on_device()
        || (row.entry.installed && matches!(row.state, ModelState::Verifying(_)))
}

fn can_run(row: &ModelRow) -> bool {
    !matches!(row.state, ModelState::Unavailable { .. })
}

/// One model of the picker, listed once whatever versions of it exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickerModel {
    /// The name without any precision, such as `Large v3 Turbo` or `Small (English)`.
    pub title: String,
    /// Every version, largest (most precise) first; never empty.
    pub versions: Vec<ModelRow>,
    key: String,
    /// The ids of the versions the search matched; every version without a search.
    matched: Vec<String>,
}

impl PickerModel {
    fn new(versions: Vec<ModelRow>, search: &str) -> Option<Self> {
        let mut versions = versions;
        versions.sort_by_key(|row| std::cmp::Reverse(row.entry.size_bytes));
        let first = versions.first()?;
        let title = if first.entry.speaker_turns {
            format!("{} · Speaker turns", first.entry.name)
        } else {
            first.entry.name.clone()
        };
        let key = format!("model:{}", first.entry.id);
        let matched = versions
            .iter()
            .filter(|row| row.entry.matches(search))
            .map(|row| row.entry.id.clone())
            .collect();
        Some(Self {
            title,
            versions,
            key,
            matched,
        })
    }

    /// The key of the model's own row in the list, distinct from every version id.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Whether model `id` is one of the versions.
    #[must_use]
    pub fn holds(&self, id: &str) -> bool {
        self.versions.iter().any(|row| row.entry.id == id)
    }

    fn group(&self) -> ModelGroup {
        self.versions
            .first()
            .map_or(ModelGroup::Other, |row| row.entry.group)
    }

    fn superseded(&self) -> bool {
        self.versions
            .first()
            .and_then(|row| row.entry.family)
            .is_some_and(|family| family.superseded())
    }

    fn stored(&self) -> bool {
        self.versions.iter().any(stored_here)
    }

    /// Whether a version is arriving or its stored files are being checked or need repair.
    fn busy(&self) -> bool {
        self.versions.iter().any(|row| match row.state {
            ModelState::Downloading(_) | ModelState::Verifying(_) | ModelState::NeedsRepair(_) => {
                true
            }
            ModelState::NotDownloaded
            | ModelState::Downloaded
            | ModelState::Failed(_)
            | ModelState::Unavailable { .. } => false,
        })
    }

    /// The version picking the model takes among those the search matched: the selected one, else
    /// one already stored here, else the most precise; `None` when no version can run here.
    #[must_use]
    pub fn preferred(&self, selected: &str) -> Option<&ModelRow> {
        let mut runnable = self
            .versions
            .iter()
            .filter(|row| can_run(row) && self.matched.contains(&row.entry.id));
        runnable
            .clone()
            .find(|row| row.entry.id == selected)
            .or_else(|| runnable.clone().find(|row| stored_here(row)))
            .or_else(|| runnable.next())
    }

    /// The model's mark: a download or problem on any version first, then whether any version is
    /// stored, then the version picking would take.
    #[must_use]
    pub fn mark(&self, selected: &str) -> Mark {
        let pressing = self
            .versions
            .iter()
            .map(Mark::of)
            .filter(|mark| mark.rank() <= Mark::Stored.rank())
            .min_by_key(|mark| mark.rank());
        pressing
            .or_else(|| self.preferred(selected).map(Mark::of))
            .or_else(|| self.versions.first().map(Mark::of))
            .unwrap_or(Mark::Download { bytes: 0 })
    }

    /// What the line says beside its versions: a download or problem on any version, else what the
    /// version picking would take still needs; nothing when it is stored, as its version shows.
    #[must_use]
    pub fn status(&self, selected: &str) -> Option<Mark> {
        self.versions
            .iter()
            .map(Mark::of)
            .filter(|mark| mark.rank() < Mark::Stored.rank())
            .min_by_key(|mark| mark.rank())
            .or_else(|| self.preferred(selected).map(Mark::of))
            .or_else(|| self.versions.first().map(Mark::of))
            .filter(|mark| *mark != Mark::Stored)
    }

    /// The versions laid out under `columns`, each in the first free column with its tag.
    #[must_use]
    pub fn slots(&self, columns: &[String]) -> Vec<Option<&ModelRow>> {
        let mut left: Vec<&ModelRow> = self.versions.iter().collect();
        columns
            .iter()
            .map(|tag| {
                let at = left
                    .iter()
                    .position(|row| version_tag(&row.entry.precision) == *tag)?;
                Some(left.remove(at))
            })
            .collect()
    }
}

/// Orders version columns: full precision first, then fewer bits further right.
fn column_rank(precision: &Precision) -> u16 {
    match precision {
        Precision::Full | Precision::Mixed => 0,
        Precision::Quantized { bits } => 256 - u16::from(*bits),
        Precision::Other(_) => 512,
    }
}

/// The version columns that fit every model of `models`, in the order the lines show them.
fn columns<'a>(models: impl Iterator<Item = &'a PickerModel>) -> Vec<String> {
    let mut all: Vec<(u16, String, usize)> = Vec::new();
    for model in models {
        let mut own: Vec<(u16, String, usize)> = Vec::new();
        for row in &model.versions {
            let tag = version_tag(&row.entry.precision);
            match own.iter_mut().find(|column| column.1 == tag) {
                Some(column) => column.2 += 1,
                None => own.push((column_rank(&row.entry.precision), tag, 1)),
            }
        }
        for (rank, tag, count) in own {
            match all.iter_mut().find(|column| column.1 == tag) {
                Some(column) => column.2 = column.2.max(count),
                None => all.push((rank, tag, count)),
            }
        }
    }
    all.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    all.into_iter()
        .flat_map(|(_, tag, count)| std::iter::repeat_n(tag, count))
        .collect()
}

/// The model picker's list for one scope and search.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PickerList {
    /// Models with a version on this device or being downloaded or checked; in
    /// [`PickerScope::Downloaded`] also the selected model.
    pub downloaded: Vec<PickerModel>,
    /// The other models; only [`PickerScope::All`] has them.
    pub available: Vec<PickerModel>,
    /// Models a newer family replaces, behind a fold; only [`PickerScope::All`] has them.
    pub older: Vec<PickerModel>,
    /// How many models matching the search the scope leaves out.
    pub hidden: usize,
}

impl PickerList {
    /// Whether nothing is listed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.downloaded.is_empty() && self.available.is_empty() && self.older.is_empty()
    }

    /// Whether the older models show without being asked: a search reached them or one is selected.
    #[must_use]
    pub fn older_opens(&self, search: &str, selected: &str) -> bool {
        !self.older.is_empty()
            && (!search.trim().is_empty() || self.older.iter().any(|model| model.holds(selected)))
    }

    /// The version columns shared by every line, so equal versions line up from line to line.
    #[must_use]
    pub fn columns(&self) -> Vec<String> {
        columns(
            self.downloaded
                .iter()
                .chain(&self.available)
                .chain(&self.older),
        )
    }
}

/// Groups `rows` into one entry per model, keeps those `scope` and `search` let through, and orders
/// them the way the Models screen does, the downloaded ones first.
#[must_use]
pub fn picker_list(
    rows: &[ModelRow],
    scope: PickerScope,
    search: &str,
    selected: &str,
) -> PickerList {
    let search = search.trim();
    let mut models: Vec<PickerModel> = cluster(rows)
        .into_iter()
        .filter_map(|cluster| PickerModel::new(cluster.rows().cloned().collect(), search))
        .filter(|model| !model.matched.is_empty())
        .collect();
    models.sort_by_key(|model| {
        ModelGroup::ORDER
            .iter()
            .position(|group| *group == model.group())
    });
    let mut list = PickerList::default();
    for model in models {
        let here = model.stored() || model.busy();
        let chosen = model.holds(selected);
        match scope {
            PickerScope::Downloaded if here || chosen => list.downloaded.push(model),
            PickerScope::Downloaded => list.hidden += 1,
            PickerScope::All if here => list.downloaded.push(model),
            PickerScope::All if model.superseded() && !chosen => list.older.push(model),
            PickerScope::All => list.available.push(model),
        }
    }
    list
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_catalog::{
        activity::{DownloadProgress, VerifyProgress},
        entry::ModelEntry,
    };
    use purr_common::platform::{ModelInfo, ModelMetadata, model_flag};

    fn row(id: &str, family: &str, size: u64, quantization: Option<&str>) -> ModelRow {
        let english = id.contains(".en");
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

    fn stored(mut row: ModelRow) -> ModelRow {
        row.entry.installed = true;
        row.state = ModelState::Downloaded;
        row
    }

    fn catalog() -> Vec<ModelRow> {
        vec![
            row("tiny", "tiny", 75, None),
            row("tiny-q5_1", "tiny", 31, Some("q5_1")),
            stored(row("base", "base", 140, None)),
            row("base.en", "base", 141, None),
            row("small", "small", 480, None),
            row("large-v2", "large-v2", 3_000, None),
            row("large-v2-q5_0", "large-v2", 1_080, Some("q5_0")),
            row("large-v3", "large-v3", 3_001, None),
            row("large-v3-turbo", "large-v3-turbo", 1_600, None),
            stored(row(
                "large-v3-turbo-q5_0",
                "large-v3-turbo",
                550,
                Some("q5_0"),
            )),
            row("large-v3-turbo-q8_0", "large-v3-turbo", 870, Some("q8_0")),
        ]
    }

    fn titles(models: &[PickerModel]) -> Vec<&str> {
        models.iter().map(|model| model.title.as_str()).collect()
    }

    fn versions(model: &PickerModel) -> Vec<&str> {
        model
            .versions
            .iter()
            .map(|row| row.entry.id.as_str())
            .collect()
    }

    #[test]
    fn a_stored_quantized_version_lists_its_model_once_under_the_plain_name() {
        let list = picker_list(&catalog(), PickerScope::Downloaded, "", "");
        assert_eq!(titles(&list.downloaded), ["Large v3 Turbo", "Base"]);
        assert_eq!(
            versions(&list.downloaded[0]),
            [
                "large-v3-turbo",
                "large-v3-turbo-q8_0",
                "large-v3-turbo-q5_0"
            ]
        );
        assert_eq!(list.downloaded[0].mark(""), Mark::Stored);
        assert_eq!(list.downloaded[0].status(""), None);
    }

    #[test]
    fn all_lists_every_model_once_the_downloaded_ones_first() {
        let list = picker_list(&catalog(), PickerScope::All, "", "");
        assert_eq!(titles(&list.downloaded), ["Large v3 Turbo", "Base"]);
        assert_eq!(
            titles(&list.available),
            ["Large v3", "Small", "Base (English)", "Tiny"]
        );
        assert_eq!(titles(&list.older), ["Large v2"]);
        assert_eq!(list.hidden, 0);
    }

    #[test]
    fn downloaded_counts_what_it_leaves_out() {
        let list = picker_list(&catalog(), PickerScope::Downloaded, "", "");
        assert_eq!(list.hidden, 5);
        assert_eq!(list.older, []);
    }

    #[test]
    fn the_selected_model_and_a_running_download_stay_in_downloaded() {
        let mut rows = catalog();
        rows[4].state = ModelState::Downloading(DownloadProgress::default());
        let list = picker_list(&rows, PickerScope::Downloaded, "", "tiny-q5_1");
        assert_eq!(
            titles(&list.downloaded),
            ["Large v3 Turbo", "Small", "Base", "Tiny"]
        );
    }

    #[test]
    fn a_stored_or_selected_older_model_is_not_folded_away() {
        let mut rows = catalog();
        rows[6] = stored(rows[6].clone());
        let list = picker_list(&rows, PickerScope::All, "", "");
        assert_eq!(list.older, []);
        assert_eq!(
            titles(&list.downloaded),
            ["Large v3 Turbo", "Large v2", "Base"]
        );
        let list = picker_list(&catalog(), PickerScope::All, "", "large-v2-q5_0");
        assert_eq!(list.older, []);
        assert_eq!(titles(&list.available)[..2], ["Large v2", "Large v3"]);
    }

    #[test]
    fn search_finds_a_model_through_any_version_and_keeps_it_whole() {
        let list = picker_list(&catalog(), PickerScope::All, " q8 ", "");
        assert_eq!(titles(&list.downloaded), ["Large v3 Turbo"]);
        assert_eq!(list.downloaded[0].versions.len(), 3);
        let list = picker_list(&catalog(), PickerScope::Downloaded, "small", "");
        assert_eq!((list.is_empty(), list.hidden), (true, 1));
    }

    #[test]
    fn the_older_models_open_for_a_search_or_when_one_is_selected() {
        let list = picker_list(&catalog(), PickerScope::All, "", "");
        assert_eq!(list.older_opens("", "base"), false);
        assert_eq!(list.older_opens("", "large-v2-q5_0"), true);
        let found = picker_list(&catalog(), PickerScope::All, "large", "");
        assert_eq!(found.older_opens("large", "base"), true);
    }

    #[test]
    fn picking_a_model_takes_the_selected_then_a_stored_then_the_most_precise_version() {
        let list = picker_list(&catalog(), PickerScope::All, "", "");
        let turbo = &list.downloaded[0];
        let id = |row: Option<&ModelRow>| row.map(|row| row.entry.id.clone());
        assert_eq!(
            id(turbo.preferred("large-v3-turbo-q8_0")),
            Some("large-v3-turbo-q8_0".to_owned())
        );
        assert_eq!(
            id(turbo.preferred("")),
            Some("large-v3-turbo-q5_0".to_owned())
        );
        let tiny = &list.available[3];
        assert_eq!(id(tiny.preferred("")), Some("tiny".to_owned()));
    }

    #[test]
    fn picking_a_model_found_through_a_version_takes_that_version() {
        let list = picker_list(&catalog(), PickerScope::All, "turbo q8", "");
        assert_eq!(
            list.downloaded[0]
                .preferred("")
                .map(|row| row.entry.id.as_str()),
            Some("large-v3-turbo-q8_0")
        );
    }

    #[test]
    fn a_model_that_cannot_run_here_offers_nothing_to_pick() {
        let mut rows = vec![row("large-v3-turbo", "large-v3-turbo", 1_600, None)];
        rows[0].state = ModelState::Unavailable {
            reason: UnavailableReason::NeedsWebGpu,
            installed: false,
        };
        let list = picker_list(&rows, PickerScope::All, "", "");
        assert_eq!(list.available[0].preferred(""), None);
        assert_eq!(
            list.available[0].mark(""),
            Mark::Unavailable(UnavailableReason::NeedsWebGpu)
        );
    }

    #[test]
    fn a_download_on_any_version_is_the_models_mark() {
        let mut rows = catalog();
        rows[1].state = ModelState::Downloading(DownloadProgress {
            loaded_bytes: 1,
            total_bytes: Some(4),
            bytes_per_second: None,
        });
        let list = picker_list(&rows, PickerScope::All, "tiny", "");
        assert_eq!(
            list.downloaded[0].mark(""),
            Mark::Downloading { percent: Some(25) }
        );
        let list = picker_list(&catalog(), PickerScope::All, "small", "");
        assert_eq!(
            list.available[0].status(""),
            Some(Mark::Download { bytes: 480 })
        );
    }

    #[test]
    fn a_stored_model_being_checked_counts_as_downloaded() {
        let mut rows = catalog();
        rows[2].state = ModelState::Verifying(VerifyProgress::default());
        assert_eq!(PickerScope::initial(&rows), PickerScope::Downloaded);
        let list = picker_list(&rows, PickerScope::Downloaded, "base", "");
        assert_eq!(list.downloaded[0].mark(""), Mark::Checking);
    }

    #[test]
    fn the_picker_opens_on_all_until_something_is_stored() {
        let rows: Vec<ModelRow> = catalog()
            .into_iter()
            .map(|mut row| {
                row.entry.installed = false;
                row.state = ModelState::NotDownloaded;
                row
            })
            .collect();
        assert_eq!(PickerScope::initial(&rows), PickerScope::All);
        assert_eq!(PickerScope::initial(&catalog()), PickerScope::Downloaded);
    }

    #[test]
    fn versions_are_named_by_precision_only() {
        assert_eq!(version_tag(&Precision::Full), "Full");
        assert_eq!(version_tag(&Precision::Mixed), "Full");
        assert_eq!(version_tag(&Precision::Quantized { bits: 5 }), "Q5");
    }

    #[test]
    fn model_keys_never_collide_with_version_ids() {
        let list = picker_list(&catalog(), PickerScope::All, "", "");
        let all: Vec<&PickerModel> = list
            .downloaded
            .iter()
            .chain(&list.available)
            .chain(&list.older)
            .collect();
        assert_eq!(
            all.iter().filter(|model| model.holds(model.key())).count(),
            0
        );
    }

    #[test]
    fn every_line_shares_the_version_columns_full_precision_first() {
        let list = picker_list(&catalog(), PickerScope::All, "", "");
        let columns = list.columns();
        assert_eq!(columns, ["Full", "Q8", "Q5"]);
        let slot_ids = |model: &PickerModel| -> Vec<Option<String>> {
            model
                .slots(&columns)
                .into_iter()
                .map(|row| row.map(|row| row.entry.id.clone()))
                .collect()
        };
        assert_eq!(
            slot_ids(&list.older[0]),
            [
                Some("large-v2".to_owned()),
                None,
                Some("large-v2-q5_0".to_owned())
            ]
        );
        assert_eq!(
            slot_ids(&list.available[1]),
            [Some("small".to_owned()), None, None]
        );
    }

    #[test]
    fn two_versions_with_one_tag_get_a_column_each() {
        let rows = vec![
            row("tiny", "tiny", 75, None),
            row("tiny-fp16", "tiny", 70, Some("fp16")),
            row("tiny-q4", "tiny", 20, Some("q4_0")),
        ];
        let list = picker_list(&rows, PickerScope::All, "", "");
        assert_eq!(list.columns(), ["Full", "Full", "Q4"]);
    }

    #[test]
    fn a_failed_download_is_not_listed_as_downloaded() {
        let mut rows = catalog();
        rows[7].state = ModelState::Failed(crate::model_catalog::activity::DownloadFailure {
            kind: purr_common::platform::ModelFailureKind::Network,
            message: String::new(),
        });
        let list = picker_list(&rows, PickerScope::All, "", "");
        assert_eq!(titles(&list.available)[..1], ["Large v3"]);
        assert_eq!(list.available[0].status(""), Some(Mark::Failed));
        let list = picker_list(&rows, PickerScope::Downloaded, "", "");
        assert_eq!(titles(&list.downloaded), ["Large v3 Turbo", "Base"]);
    }

    #[test]
    fn a_search_for_a_missing_version_of_a_stored_model_shows_its_download() {
        let list = picker_list(&catalog(), PickerScope::All, "turbo q8", "");
        assert_eq!(
            list.downloaded[0].status(""),
            Some(Mark::Download { bytes: 870 })
        );
    }
}
