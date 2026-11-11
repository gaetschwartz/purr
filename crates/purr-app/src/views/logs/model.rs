use super::{
    filter::{Entries, LevelFilter, LogFilter},
    window::RowWindow,
};
use std::{ops::Range, rc::Rc};

/// Everything the log list shows and how: the captured entries, the filter, and the rows on screen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogsModel {
    entries: Entries,
    matching: Entries,
    level: LevelFilter,
    query: String,
    filter: LogFilter,
    window: RowWindow,
    following: bool,
}

impl LogsModel {
    /// A model of `entries` with no filter, following the newest entry.
    #[must_use]
    pub fn new(entries: Entries) -> Self {
        Self {
            matching: Rc::clone(&entries),
            entries,
            level: LevelFilter::All,
            query: String::new(),
            filter: LogFilter::default(),
            window: RowWindow::default(),
            following: true,
        }
    }

    /// Takes the latest captured entries; returns how many new entries pass the filter.
    pub fn refresh(&mut self, entries: Entries) -> usize {
        let newest_seen = self.entries.last().map_or(0, |entry| entry.id);
        let fresh = self.filter.count_newer(&entries, newest_seen);
        self.matching = self.filter.apply(&entries);
        self.entries = entries;
        fresh
    }

    /// Lists only what passes `level` and `query`, starting again from the newest entries.
    pub fn set_filter(&mut self, level: LevelFilter, query: String) {
        self.filter = LogFilter::new(level, &query);
        self.level = level;
        self.query = query;
        self.matching = self.filter.apply(&self.entries);
        self.window = RowWindow::default().pinned(&self.matching, !self.following);
    }

    /// Sets whether the list follows the newest entry, which decides whether its rows may slide.
    pub fn set_following(&mut self, following: bool) {
        self.following = following;
        self.window = self.window.pinned(&self.matching, !following);
    }

    /// Adds a step of earlier entries to the rows shown.
    pub fn show_earlier(&mut self) {
        self.window = self.window.earlier(&self.matching);
    }

    /// The level threshold in force.
    #[must_use]
    pub const fn level(&self) -> LevelFilter {
        self.level
    }

    /// The search text in force.
    #[must_use]
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Every entry captured.
    #[must_use]
    pub fn total(&self) -> usize {
        self.entries.len()
    }

    /// The entries that pass the filter.
    #[must_use]
    pub const fn matching(&self) -> &Entries {
        &self.matching
    }

    /// The positions in [`Self::matching`] that are on screen.
    #[must_use]
    pub fn rows(&self) -> Range<usize> {
        self.window.range(&self.matching)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::logging::{LogEntry, SerdeTracingLevel};
    use std::sync::Arc;

    fn entry(id: u64, level: SerdeTracingLevel, message: &str) -> Arc<LogEntry> {
        Arc::new(LogEntry {
            id,
            timestamp: 0,
            level,
            target: "app".to_owned(),
            message: message.to_owned(),
            fields: vec![],
            file: None,
            line: None,
        })
    }

    fn numbered(range: std::ops::RangeInclusive<u64>) -> Entries {
        Rc::new(
            range
                .map(|id| {
                    let level = if id % 10 == 0 {
                        SerdeTracingLevel::ERROR
                    } else {
                        SerdeTracingLevel::INFO
                    };
                    entry(id, level, &format!("message {id}"))
                })
                .collect(),
        )
    }

    fn visible_ids(model: &LogsModel) -> Vec<u64> {
        model
            .matching()
            .get(model.rows())
            .unwrap_or_default()
            .iter()
            .map(|entry| entry.id)
            .collect()
    }

    #[test]
    fn refreshing_counts_the_new_entries_that_pass_the_filter() {
        let mut model = LogsModel::new(numbered(1..=20));
        assert_eq!(model.refresh(numbered(1..=35)), 15);

        model.set_filter(
            LevelFilter::AtLeast(SerdeTracingLevel::ERROR),
            String::new(),
        );
        assert_eq!(model.refresh(numbered(1..=62)), 3);
        assert_eq!(model.total(), 62);
        assert_eq!(model.matching().len(), 6);
    }

    #[test]
    fn a_following_list_shows_the_newest_rows_as_entries_arrive() {
        let mut model = LogsModel::new(numbered(1..=1000));
        assert_eq!(model.rows().start, 600);
        assert_eq!(model.refresh(numbered(1..=1010)), 10);
        assert_eq!(model.rows(), 610..1010);
        assert_eq!(model.rows().start, 610);
    }

    #[test]
    fn a_list_that_stopped_following_keeps_its_rows_in_place() {
        let mut model = LogsModel::new(numbered(1..=1000));
        model.set_following(false);
        assert_eq!(model.refresh(numbered(1..=1010)), 10);
        assert_eq!(model.rows(), 600..1010);

        model.set_following(true);
        assert_eq!(model.rows(), 610..1010);
    }

    #[test]
    fn changing_the_filter_starts_again_from_the_newest_matches() {
        let mut model = LogsModel::new(numbered(1..=1000));
        model.show_earlier();
        model.set_filter(
            LevelFilter::AtLeast(SerdeTracingLevel::ERROR),
            String::new(),
        );
        assert_eq!(model.matching().len(), 100);
        assert_eq!(model.rows(), 0..100);
        assert_eq!(
            model.level(),
            LevelFilter::AtLeast(SerdeTracingLevel::ERROR)
        );

        model.set_filter(LevelFilter::All, "message 99".to_owned());
        assert_eq!(
            visible_ids(&model),
            vec![99, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999]
        );
        assert_eq!(model.query(), "message 99");
    }

    #[test]
    fn showing_earlier_adds_a_step_of_rows_above() {
        let mut model = LogsModel::new(numbered(1..=1000));
        model.show_earlier();
        assert_eq!(model.rows(), 200..1000);
        assert_eq!(model.rows().start, 200);
    }

    #[test]
    fn clearing_the_entries_empties_the_list() {
        let mut model = LogsModel::new(numbered(1..=50));
        assert_eq!(model.refresh(Rc::new(Vec::new())), 0);
        assert_eq!(model.total(), 0);
        assert_eq!(visible_ids(&model), Vec::<u64>::new());
    }
}
