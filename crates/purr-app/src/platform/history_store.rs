//! The desktop's history: one JSON file per entry in a directory, written atomically.

use purr_common::{
    history::{
        entry::{AudioAvailability, HistoryEntry, HistoryId, HistoryRecord, HistorySource},
        envelope::{self, UnreadableReason},
        listing::{HistoryListing, Saved, UnreadableEntry},
        retention::{Limits, Stored, evictions},
        title::{TitleProblem, clean_title},
    },
    platform::PlatformError,
};
use std::{
    fs,
    io::{self, Write as _},
    path::{Path, PathBuf},
};

const ENTRY_EXTENSION: &str = "json";
const PARTIAL_EXTENSION: &str = "tmp";

/// Why a history operation on disk failed.
#[derive(Debug, thiserror::Error)]
pub enum HistoryDirError {
    #[error("{action} {path}: {source}")]
    Io {
        action: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("There is no saved transcription {0}")]
    NotFound(HistoryId),
    #[error("The saved transcription {id} can't be read: {reason}")]
    Unreadable {
        id: HistoryId,
        reason: UnreadableReason,
    },
    #[error("The transcription could not be written: {0}")]
    Encode(#[from] serde_json::Error),
    #[error("{0}")]
    Title(#[from] TitleProblem),
}

impl From<HistoryDirError> for PlatformError {
    fn from(error: HistoryDirError) -> Self {
        match error {
            HistoryDirError::NotFound(id) => Self::HistoryEntryNotFound { id },
            HistoryDirError::Io { .. }
            | HistoryDirError::Unreadable { .. }
            | HistoryDirError::Encode(_)
            | HistoryDirError::Title(_) => Self::history(error),
        }
    }
}

fn io_error(action: &'static str, path: &Path) -> impl FnOnce(io::Error) -> HistoryDirError {
    let path = path.to_path_buf();
    move |source| HistoryDirError::Io {
        action,
        path,
        source,
    }
}

fn replace_file(partial: &Path, target: &Path, content: &[u8]) -> io::Result<()> {
    let mut file = fs::File::create(partial)?;
    file.write_all(content)?;
    file.sync_all()?;
    drop(file);
    fs::rename(partial, target)
}

/// A directory of saved transcriptions.
#[derive(Debug, Clone)]
pub struct HistoryDir {
    dir: PathBuf,
    limits: Limits,
}

impl HistoryDir {
    #[must_use]
    pub const fn new(dir: PathBuf, limits: Limits) -> Self {
        Self { dir, limits }
    }

    fn entry_path(&self, id: &HistoryId) -> PathBuf {
        self.dir.join(format!("{id}.{ENTRY_EXTENSION}"))
    }

    fn files(&self, extension: &str) -> Result<Vec<PathBuf>, HistoryDirError> {
        let listing = match fs::read_dir(&self.dir) {
            Ok(listing) => listing,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(io_error("Could not read", &self.dir)(error)),
        };
        let mut found = Vec::new();
        for item in listing {
            let item = item.map_err(io_error("Could not read", &self.dir))?;
            let path = item.path();
            if path.extension().is_some_and(|found| found == extension) && path.is_file() {
                found.push(path);
            }
        }
        Ok(found)
    }

    fn read(path: &Path) -> Result<HistoryEntry, UnreadableReason> {
        let text = fs::read_to_string(path).map_err(|error| UnreadableReason::Corrupt {
            detail: error.to_string(),
        })?;
        let entry = envelope::decode(&text)?;
        let stem = path.file_stem().and_then(|stem| stem.to_str());
        if stem == Some(entry.id.as_str()) {
            Ok(entry)
        } else {
            Err(UnreadableReason::Corrupt {
                detail: "its id differs from its file name".to_owned(),
            })
        }
    }

    fn name_of(path: &Path) -> String {
        path.file_name()
            .map_or_else(String::new, |name| name.to_string_lossy().into_owned())
    }

    fn availability(entry: &HistoryEntry) -> AudioAvailability {
        match &entry.source {
            HistorySource::NotKept => AudioAvailability::NotKept,
            HistorySource::Path { path } if path.is_file() => AudioAvailability::Available,
            HistorySource::Path { .. } => AudioAvailability::Missing,
        }
    }

    /// Every readable entry as a summary, newest first, and the files that could not be read.
    pub fn list(&self) -> Result<HistoryListing, HistoryDirError> {
        let mut summaries = Vec::new();
        let mut unreadable = Vec::new();
        for path in self.files(ENTRY_EXTENSION)? {
            match Self::read(&path) {
                Ok(entry) => summaries.push(entry.summary(Self::availability(&entry))),
                Err(reason) => unreadable.push(UnreadableEntry {
                    name: Self::name_of(&path),
                    reason,
                }),
            }
        }
        Ok(HistoryListing::new(summaries, unreadable))
    }

    /// The entry `id`, with whether its audio is still where it was.
    pub fn get(&self, id: &HistoryId) -> Result<HistoryRecord, HistoryDirError> {
        let path = self.entry_path(id);
        if !path.is_file() {
            return Err(HistoryDirError::NotFound(id.clone()));
        }
        let entry = Self::read(&path).map_err(|reason| HistoryDirError::Unreadable {
            id: id.clone(),
            reason,
        })?;
        let audio = Self::availability(&entry);
        Ok(HistoryRecord { entry, audio })
    }

    fn write(&self, entry: &HistoryEntry) -> Result<(), HistoryDirError> {
        fs::create_dir_all(&self.dir).map_err(io_error("Could not create", &self.dir))?;
        let text = envelope::encode(entry)?;
        let target = self.entry_path(&entry.id);
        let partial = target.with_extension(format!("{ENTRY_EXTENSION}.{PARTIAL_EXTENSION}"));
        if let Err(error) = replace_file(&partial, &target, text.as_bytes()) {
            if let Err(cleanup) = fs::remove_file(&partial)
                && cleanup.kind() != io::ErrorKind::NotFound
            {
                tracing::warn!(
                    "Could not remove the partial file {}: {cleanup}",
                    partial.display()
                );
            }
            return Err(io_error("Could not write", &target)(error));
        }
        Ok(())
    }

    /// Stores `entry`, then drops the oldest entries beyond the limits.
    pub fn save(&self, entry: &HistoryEntry) -> Result<Saved, HistoryDirError> {
        self.write(entry)?;
        let mut stored = Vec::new();
        for path in self.files(ENTRY_EXTENSION)? {
            let Ok(read) = Self::read(&path) else {
                continue;
            };
            let bytes = fs::metadata(&path)
                .map_err(io_error("Could not measure", &path))?
                .len();
            stored.push(Stored {
                id: read.id,
                created_at_ms: read.created_at_ms,
                bytes,
            });
        }
        let evicted = evictions(&stored, self.limits);
        for id in &evicted {
            let path = self.entry_path(id);
            fs::remove_file(&path).map_err(io_error("Could not remove", &path))?;
        }
        Ok(Saved { evicted })
    }

    /// Gives the entry `id` the title `title`.
    pub fn rename(&self, id: &HistoryId, title: &str) -> Result<(), HistoryDirError> {
        let mut record = self.get(id)?;
        record.entry.title = clean_title(title)?;
        self.write(&record.entry)
    }

    /// Deletes the entry `id`.
    pub fn delete(&self, id: &HistoryId) -> Result<(), HistoryDirError> {
        let path = self.entry_path(id);
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                Err(HistoryDirError::NotFound(id.clone()))
            }
            Err(error) => Err(io_error("Could not remove", &path)(error)),
        }
    }

    /// Deletes every stored entry, readable or not, and any leftover partial file.
    pub fn clear(&self) -> Result<usize, HistoryDirError> {
        let entries = self.files(ENTRY_EXTENSION)?;
        for path in entries.iter().chain(&self.files(PARTIAL_EXTENSION)?) {
            fs::remove_file(path).map_err(io_error("Could not remove", path))?;
        }
        Ok(entries.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::{history::entry::LanguageOrigin, platform::TranscriptSegment};

    const SMALL: Limits = Limits {
        max_entries: 3,
        max_bytes: 1_000_000,
    };

    fn entry(name: &str, created_at_ms: i64) -> HistoryEntry {
        HistoryEntry {
            id: HistoryId::parse(name).unwrap(),
            title: format!("{name}.ogg"),
            file_name: format!("{name}.ogg"),
            created_at_ms,
            audio_seconds: 3.0,
            language: Some("fr".to_owned()),
            language_origin: LanguageOrigin::Detected,
            model_id: "tiny".to_owned(),
            model_label: "Tiny".to_owned(),
            processing_seconds: 1.0,
            preview: "Bonjour".to_owned(),
            segments: vec![TranscriptSegment {
                text: format!("Bonjour {name}"),
                start_time: 0.0,
                end_time: 2.0,
                words: Vec::new(),
            }],
            peaks: vec![10, 200],
            source: HistorySource::NotKept,
        }
    }

    fn dir(limits: Limits) -> (tempfile::TempDir, HistoryDir) {
        let temp = tempfile::tempdir().unwrap();
        let history = HistoryDir::new(temp.path().join("history"), limits);
        (temp, history)
    }

    fn titles(listing: &HistoryListing) -> Vec<&str> {
        listing
            .summaries
            .iter()
            .map(|summary| summary.title.as_str())
            .collect()
    }

    #[test]
    fn a_directory_that_does_not_exist_yet_is_an_empty_history() {
        let (_temp, history) = dir(SMALL);
        assert_eq!(history.list().unwrap(), HistoryListing::default());
    }

    #[test]
    fn saved_entries_come_back_newest_first_and_whole() {
        let (_temp, history) = dir(SMALL);
        history.save(&entry("a", 1)).unwrap();
        history.save(&entry("b", 2)).unwrap();
        let listing = history.list().unwrap();
        assert_eq!(titles(&listing), vec!["b.ogg", "a.ogg"]);
        let record = history.get(&entry("a", 1).id).unwrap();
        assert_eq!(record.entry, entry("a", 1));
        assert_eq!(record.audio, AudioAvailability::NotKept);
    }

    #[test]
    fn saving_the_same_id_replaces_it() {
        let (_temp, history) = dir(SMALL);
        history.save(&entry("a", 1)).unwrap();
        let mut changed = entry("a", 1);
        changed.title = "Changed".to_owned();
        history.save(&changed).unwrap();
        assert_eq!(titles(&history.list().unwrap()), vec!["Changed"]);
    }

    #[test]
    fn a_save_leaves_no_partial_file_behind() {
        let (_temp, history) = dir(SMALL);
        history.save(&entry("a", 1)).unwrap();
        assert_eq!(
            history.files(PARTIAL_EXTENSION).unwrap(),
            Vec::<PathBuf>::new()
        );
    }

    #[test]
    fn a_partial_file_is_not_an_entry_and_is_not_reported() {
        let (_temp, history) = dir(SMALL);
        history.save(&entry("a", 1)).unwrap();
        fs::write(history.dir.join("b.json.tmp"), "{\"vers").unwrap();
        let listing = history.list().unwrap();
        assert_eq!(listing.summaries.len(), 1);
        assert_eq!(listing.unreadable, Vec::new());
    }

    #[test]
    fn a_damaged_file_is_reported_and_the_rest_still_lists() {
        let (_temp, history) = dir(SMALL);
        history.save(&entry("a", 1)).unwrap();
        fs::write(history.dir.join("broken.json"), "not json").unwrap();
        fs::write(
            history.dir.join("future.json"),
            "{\"version\":7,\"entry\":{}}",
        )
        .unwrap();
        let mut listing = history.list().unwrap();
        assert_eq!(titles(&listing), vec!["a.ogg"]);
        listing.unreadable.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(
            listing
                .unreadable
                .iter()
                .map(|entry| (
                    entry.name.as_str(),
                    matches!(entry.reason, UnreadableReason::NewerVersion { found: 7 })
                ))
                .collect::<Vec<_>>(),
            vec![("broken.json", false), ("future.json", true)]
        );
    }

    #[test]
    fn an_entry_whose_id_differs_from_its_file_name_is_unreadable() {
        let (_temp, history) = dir(SMALL);
        history.save(&entry("a", 1)).unwrap();
        fs::copy(history.dir.join("a.json"), history.dir.join("b.json")).unwrap();
        let listing = history.list().unwrap();
        assert_eq!(listing.summaries.len(), 1);
        assert_eq!(listing.unreadable.len(), 1);
    }

    #[test]
    fn the_count_cap_drops_the_oldest_on_save() {
        let (_temp, history) = dir(SMALL);
        for (name, at) in [("a", 1), ("b", 2), ("c", 3)] {
            assert_eq!(history.save(&entry(name, at)).unwrap().evicted, Vec::new());
        }
        let saved = history.save(&entry("d", 4)).unwrap();
        assert_eq!(saved.evicted, vec![entry("a", 1).id]);
        assert_eq!(
            titles(&history.list().unwrap()),
            vec!["d.ogg", "c.ogg", "b.ogg"]
        );
    }

    #[test]
    fn the_size_cap_drops_the_oldest_on_save() {
        let one = envelope::encode(&entry("a", 1)).unwrap().len() as u64;
        let (_temp, history) = dir(Limits {
            max_entries: 100,
            max_bytes: one * 2 + one / 2,
        });
        history.save(&entry("a", 1)).unwrap();
        history.save(&entry("b", 2)).unwrap();
        let saved = history.save(&entry("c", 3)).unwrap();
        assert_eq!(saved.evicted, vec![entry("a", 1).id]);
        assert_eq!(titles(&history.list().unwrap()), vec!["c.ogg", "b.ogg"]);
    }

    #[test]
    fn restoring_an_older_entry_into_a_full_history_evicts_it_again() {
        let (_temp, history) = dir(SMALL);
        for (name, at) in [("b", 2), ("c", 3), ("d", 4)] {
            history.save(&entry(name, at)).unwrap();
        }
        let saved = history.save(&entry("a", 1)).unwrap();
        assert_eq!(saved.evicted, vec![entry("a", 1).id]);
    }

    #[test]
    fn rename_validates_and_keeps_the_rest() {
        let (_temp, history) = dir(SMALL);
        history.save(&entry("a", 1)).unwrap();
        let id = entry("a", 1).id;
        history.rename(&id, "  Réunion   du lundi ").unwrap();
        let record = history.get(&id).unwrap();
        assert_eq!(record.entry.title, "Réunion du lundi");
        assert_eq!(record.entry.segments, entry("a", 1).segments);
        assert!(matches!(
            history.rename(&id, "   "),
            Err(HistoryDirError::Title(TitleProblem::Empty))
        ));
        assert_eq!(history.get(&id).unwrap().entry.title, "Réunion du lundi");
    }

    #[test]
    fn delete_removes_one_and_reports_a_missing_one() {
        let (_temp, history) = dir(SMALL);
        history.save(&entry("a", 1)).unwrap();
        history.save(&entry("b", 2)).unwrap();
        history.delete(&entry("a", 1).id).unwrap();
        assert_eq!(titles(&history.list().unwrap()), vec!["b.ogg"]);
        assert!(matches!(
            history.delete(&entry("a", 1).id),
            Err(HistoryDirError::NotFound(_))
        ));
        assert!(matches!(
            history.get(&entry("a", 1).id),
            Err(HistoryDirError::NotFound(_))
        ));
    }

    #[test]
    fn an_undone_delete_puts_the_entry_back_where_it_was() {
        let (_temp, history) = dir(SMALL);
        for (name, at) in [("a", 1), ("b", 2), ("c", 3)] {
            history.save(&entry(name, at)).unwrap();
        }
        let record = history.get(&entry("b", 2).id).unwrap();
        history.delete(&record.entry.id).unwrap();
        history.save(&record.entry).unwrap();
        assert_eq!(
            titles(&history.list().unwrap()),
            vec!["c.ogg", "b.ogg", "a.ogg"]
        );
    }

    #[test]
    fn clear_removes_unreadable_files_too() {
        let (_temp, history) = dir(SMALL);
        history.save(&entry("a", 1)).unwrap();
        fs::write(history.dir.join("broken.json"), "x").unwrap();
        assert_eq!(history.clear().unwrap(), 2);
        assert_eq!(history.list().unwrap(), HistoryListing::default());
    }

    #[test]
    fn audio_is_available_only_while_the_original_file_exists() {
        let (temp, history) = dir(SMALL);
        let audio = temp.path().join("memo.ogg");
        fs::write(&audio, b"audio").unwrap();
        let mut kept = entry("a", 1);
        kept.source = HistorySource::Path {
            path: audio.clone(),
        };
        history.save(&kept).unwrap();
        assert_eq!(
            history.get(&kept.id).unwrap().audio,
            AudioAvailability::Available
        );
        assert_eq!(
            history.list().unwrap().summaries[0].audio,
            AudioAvailability::Available
        );
        fs::remove_file(&audio).unwrap();
        assert_eq!(
            history.get(&kept.id).unwrap().audio,
            AudioAvailability::Missing
        );
        assert_eq!(
            history.list().unwrap().summaries[0].audio,
            AudioAvailability::Missing
        );
    }
}
