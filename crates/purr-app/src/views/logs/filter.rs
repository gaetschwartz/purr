use crate::platform::logging::{LogEntry, SerdeTracingLevel};
use std::{rc::Rc, sync::Arc};

/// Entries shared between the view's parts without copying them.
pub type Entries = Rc<Vec<Arc<LogEntry>>>;

const fn severity(level: SerdeTracingLevel) -> u8 {
    match level {
        SerdeTracingLevel::ERROR => 0,
        SerdeTracingLevel::WARN => 1,
        SerdeTracingLevel::INFO => 2,
        SerdeTracingLevel::DEBUG => 3,
        SerdeTracingLevel::TRACE => 4,
    }
}

/// The least severe level a row may have to be listed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LevelFilter {
    #[default]
    All,
    AtLeast(SerdeTracingLevel),
}

impl LevelFilter {
    /// The choices of the level control, in order.
    pub const CHOICES: [(Self, &'static str); 6] = [
        (Self::All, "All"),
        (Self::AtLeast(SerdeTracingLevel::ERROR), "Errors"),
        (Self::AtLeast(SerdeTracingLevel::WARN), "Warnings"),
        (Self::AtLeast(SerdeTracingLevel::INFO), "Info"),
        (Self::AtLeast(SerdeTracingLevel::DEBUG), "Debug"),
        (Self::AtLeast(SerdeTracingLevel::TRACE), "Trace"),
    ];

    /// Whether an entry of `level` is listed.
    #[must_use]
    pub const fn admits(self, level: SerdeTracingLevel) -> bool {
        match self {
            Self::All => true,
            Self::AtLeast(least) => severity(level) <= severity(least),
        }
    }
}

/// What the user asked to see: a level threshold and a search text.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LogFilter {
    level: LevelFilter,
    needle: String,
}

impl LogFilter {
    /// A filter for `level` and `query`; the query ignores case and surrounding spaces.
    #[must_use]
    pub fn new(level: LevelFilter, query: &str) -> Self {
        Self {
            level,
            needle: query.trim().to_lowercase(),
        }
    }

    /// Whether the filter lists every entry.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.level == LevelFilter::All && self.needle.is_empty()
    }

    /// Whether `entry` passes the level threshold and its target, message or fields hold the query.
    #[must_use]
    pub fn matches(&self, entry: &LogEntry) -> bool {
        self.level.admits(entry.level)
            && (self.needle.is_empty()
                || has_text(&entry.message, &self.needle)
                || has_text(&entry.target, &self.needle)
                || entry.fields.iter().any(|(key, value)| {
                    has_text(key, &self.needle) || has_text(value, &self.needle)
                }))
    }

    /// The entries of `entries` that pass, in their order.
    #[must_use]
    pub fn apply(&self, entries: &Entries) -> Entries {
        if self.is_open() {
            return Rc::clone(entries);
        }
        Rc::new(
            entries
                .iter()
                .filter(|entry| self.matches(entry))
                .map(Arc::clone)
                .collect(),
        )
    }

    /// How many of the entries numbered above `after` pass.
    #[must_use]
    pub fn count_newer(&self, entries: &[Arc<LogEntry>], after: u64) -> usize {
        entries
            .iter()
            .rev()
            .take_while(|entry| entry.id > after)
            .filter(|entry| self.matches(entry))
            .count()
    }
}

fn has_text(haystack: &str, lowercase_needle: &str) -> bool {
    haystack.to_lowercase().contains(lowercase_needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: u64, level: SerdeTracingLevel, target: &str, message: &str) -> Arc<LogEntry> {
        Arc::new(LogEntry {
            id,
            timestamp: 0,
            level,
            target: target.to_owned(),
            message: message.to_owned(),
            fields: vec![("device".to_owned(), "GPU-0".to_owned())],
            file: None,
            line: None,
        })
    }

    fn sample() -> Entries {
        Rc::new(vec![
            entry(1, SerdeTracingLevel::TRACE, "whisper", "Segment 0"),
            entry(
                2,
                SerdeTracingLevel::DEBUG,
                "audio",
                "Resampling to 16000 Hz",
            ),
            entry(
                3,
                SerdeTracingLevel::INFO,
                "purr_app::models",
                "Model Loaded",
            ),
            entry(4, SerdeTracingLevel::WARN, "gpu", "Vulkan device skipped"),
            entry(5, SerdeTracingLevel::ERROR, "models", "Download failed"),
        ])
    }

    fn ids(entries: &Entries) -> Vec<u64> {
        entries.iter().map(|entry| entry.id).collect()
    }

    #[test]
    fn a_level_lists_that_level_and_the_more_severe_ones() {
        let warnings = LogFilter::new(LevelFilter::AtLeast(SerdeTracingLevel::WARN), "");
        assert_eq!(ids(&warnings.apply(&sample())), vec![4, 5]);

        let everything = LogFilter::new(LevelFilter::AtLeast(SerdeTracingLevel::TRACE), "");
        assert_eq!(ids(&everything.apply(&sample())), vec![1, 2, 3, 4, 5]);

        let errors = LogFilter::new(LevelFilter::AtLeast(SerdeTracingLevel::ERROR), "");
        assert_eq!(ids(&errors.apply(&sample())), vec![5]);
    }

    #[test]
    fn the_search_ignores_case_and_surrounding_spaces() {
        let filter = LogFilter::new(LevelFilter::All, "  MODEL ");
        assert_eq!(ids(&filter.apply(&sample())), vec![3, 5]);
    }

    #[test]
    fn the_search_looks_in_the_target_and_the_fields() {
        assert_eq!(
            ids(&LogFilter::new(LevelFilter::All, "purr_app").apply(&sample())),
            vec![3]
        );
        assert_eq!(
            ids(&LogFilter::new(LevelFilter::All, "gpu-0").apply(&sample())),
            vec![1, 2, 3, 4, 5]
        );
        assert_eq!(
            ids(&LogFilter::new(LevelFilter::All, "nothing like this").apply(&sample())),
            Vec::<u64>::new()
        );
    }

    #[test]
    fn level_and_search_both_have_to_match() {
        let filter = LogFilter::new(LevelFilter::AtLeast(SerdeTracingLevel::WARN), "model");
        assert_eq!(ids(&filter.apply(&sample())), vec![5]);
    }

    #[test]
    fn an_open_filter_shares_the_entries_instead_of_copying_them() {
        let entries = sample();
        let filter = LogFilter::default();
        assert!(filter.is_open());
        assert!(Rc::ptr_eq(&entries, &filter.apply(&entries)));
        assert!(!LogFilter::new(LevelFilter::All, "x").is_open());
    }

    #[test]
    fn newer_entries_are_counted_after_the_last_seen_id() {
        let entries = sample();
        let open = LogFilter::default();
        assert_eq!(open.count_newer(&entries, 3), 2);
        assert_eq!(open.count_newer(&entries, 5), 0);
        assert_eq!(open.count_newer(&entries, 0), 5);

        let errors = LogFilter::new(LevelFilter::AtLeast(SerdeTracingLevel::ERROR), "");
        assert_eq!(errors.count_newer(&entries, 3), 1);
    }
}
