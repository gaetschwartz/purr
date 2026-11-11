use crate::platform::logging::LogEntry;
use std::{ops::Range, sync::Arc};

/// How many rows the list shows at first, and how many more "Show earlier" adds.
pub const ROW_STEP: usize = 400;

/// Which rows of the matching entries are on screen.
///
/// While following, the window is the newest `limit` entries and slides as new ones arrive.
/// Once following stops it is pinned to the entry it starts at, so the rows being read stay put
/// and new entries only extend it at the end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowWindow {
    limit: usize,
    pinned_from: Option<u64>,
}

impl Default for RowWindow {
    fn default() -> Self {
        Self {
            limit: ROW_STEP,
            pinned_from: None,
        }
    }
}

impl RowWindow {
    /// The positions in `entries` that are shown.
    #[must_use]
    pub fn range(&self, entries: &[Arc<LogEntry>]) -> Range<usize> {
        let len = entries.len();
        let start = self.pinned_from.map_or_else(
            || len.saturating_sub(self.limit),
            |id| {
                entries
                    .iter()
                    .position(|entry| entry.id >= id)
                    .unwrap_or(len)
            },
        );
        start..len
    }

    /// Pins the window to its current first entry, or lets it slide again.
    #[must_use]
    pub fn pinned(self, entries: &[Arc<LogEntry>], pinned: bool) -> Self {
        let pinned_from = if pinned {
            entries.get(self.range(entries).start).map(|entry| entry.id)
        } else {
            None
        };
        Self {
            pinned_from,
            ..self
        }
    }

    /// Shows [`ROW_STEP`] more entries before the current first one.
    #[must_use]
    pub fn earlier(self, entries: &[Arc<LogEntry>]) -> Self {
        let start = self.range(entries).start;
        let pinned_from = self.pinned_from.and_then(|_| {
            entries
                .get(start.saturating_sub(ROW_STEP))
                .map(|entry| entry.id)
        });
        Self {
            limit: self.limit.saturating_add(ROW_STEP),
            pinned_from,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::logging::SerdeTracingLevel;

    fn entries(ids: impl IntoIterator<Item = u64>) -> Vec<Arc<LogEntry>> {
        ids.into_iter()
            .map(|id| {
                Arc::new(LogEntry {
                    id,
                    timestamp: 0,
                    level: SerdeTracingLevel::INFO,
                    target: String::new(),
                    message: String::new(),
                    fields: vec![],
                    file: None,
                    line: None,
                })
            })
            .collect()
    }

    #[test]
    fn a_short_list_is_shown_whole() {
        let list = entries(1..=10);
        assert_eq!(RowWindow::default().range(&list), 0..10);
        assert_eq!(RowWindow::default().range(&list).start, 0);
    }

    #[test]
    fn a_following_window_slides_over_the_newest_entries() {
        let window = RowWindow::default();
        let list = entries(1..=1000);
        assert_eq!(window.range(&list), 600..1000);
        let longer = entries(1..=1003);
        assert_eq!(window.range(&longer), 603..1003);
        assert_eq!(window.range(&longer).start, 603);
    }

    #[test]
    fn a_pinned_window_keeps_its_first_row_while_entries_arrive() {
        let list = entries(1..=1000);
        let window = RowWindow::default().pinned(&list, true);
        assert_eq!(window.range(&list), 600..1000);

        let longer = entries(1..=1050);
        assert_eq!(window.range(&longer), 600..1050);
    }

    #[test]
    fn a_pinned_window_starts_at_the_oldest_entry_left_when_its_own_is_gone() {
        let list = entries(1..=1000);
        let window = RowWindow::default().pinned(&list, true);
        let trimmed = entries(701..=1100);
        assert_eq!(window.range(&trimmed), 0..400);
    }

    #[test]
    fn unpinning_makes_the_window_slide_again() {
        let list = entries(1..=1000);
        let window = RowWindow::default()
            .pinned(&list, true)
            .pinned(&list, false);
        let longer = entries(1..=1050);
        assert_eq!(window.range(&longer), 650..1050);
    }

    #[test]
    fn showing_earlier_entries_extends_the_window_at_its_start() {
        let list = entries(1..=1000);
        let sliding = RowWindow::default().earlier(&list);
        assert_eq!(sliding.range(&list), 200..1000);

        let pinned = RowWindow::default().pinned(&list, true).earlier(&list);
        assert_eq!(pinned.range(&list), 200..1000);
        let longer = entries(1..=1010);
        assert_eq!(pinned.range(&longer), 200..1010);
    }

    #[test]
    fn showing_earlier_stops_at_the_first_entry() {
        let list = entries(1..=500);
        let window = RowWindow::default().earlier(&list).earlier(&list);
        assert_eq!(window.range(&list), 0..500);
    }

    #[test]
    fn an_empty_list_has_an_empty_window() {
        let list = entries([]);
        let window = RowWindow::default().pinned(&list, true);
        assert_eq!(window.range(&list), 0..0);
    }
}
