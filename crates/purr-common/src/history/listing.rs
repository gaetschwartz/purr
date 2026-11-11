//! What listing the history returns: the entries that could be read and the ones that could not.

use super::{
    entry::{HistoryId, HistorySummary},
    envelope::UnreadableReason,
};
use std::cmp::Reverse;

/// A stored entry that could not be read; the rest of the list is unaffected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnreadableEntry {
    /// The file name or storage key it was found under.
    pub name: String,
    pub reason: UnreadableReason,
}

/// The saved transcriptions, newest first, and what could not be read.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HistoryListing {
    pub summaries: Vec<HistorySummary>,
    pub unreadable: Vec<UnreadableEntry>,
}

impl HistoryListing {
    /// A listing of `summaries` in any order, which come out newest first.
    #[must_use]
    pub fn new(mut summaries: Vec<HistorySummary>, unreadable: Vec<UnreadableEntry>) -> Self {
        summaries.sort_by_key(|summary| Reverse((summary.created_at_ms, summary.id.clone())));
        Self {
            summaries,
            unreadable,
        }
    }
}

impl HistoryListing {
    /// The listing without the entry `id`.
    #[must_use]
    pub fn removed(&self, id: &HistoryId) -> Self {
        Self {
            summaries: self
                .summaries
                .iter()
                .filter(|summary| &summary.id != id)
                .cloned()
                .collect(),
            unreadable: self.unreadable.clone(),
        }
    }

    /// The listing with the entry `id` titled `title`.
    #[must_use]
    pub fn retitled(&self, id: &HistoryId, title: &str) -> Self {
        let mut listing = self.clone();
        for summary in &mut listing.summaries {
            if &summary.id == id {
                title.clone_into(&mut summary.title);
            }
        }
        listing
    }
}

/// What saving an entry did besides storing it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Saved {
    /// The entries dropped to stay within the retention limits, oldest first.
    pub evicted: Vec<HistoryId>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::entry::{AudioAvailability, HistoryId, LanguageOrigin};

    fn summary(id: &str, created_at_ms: i64) -> HistorySummary {
        HistorySummary {
            id: HistoryId::parse(id).unwrap_or_else(|_| unreachable!()),
            title: id.to_owned(),
            file_name: id.to_owned(),
            created_at_ms,
            audio_seconds: 1.0,
            language: None,
            language_origin: LanguageOrigin::Chosen,
            model_id: "tiny".to_owned(),
            model_label: "Tiny".to_owned(),
            processing_seconds: 1.0,
            preview: String::new(),
            segment_count: 0,
            search_text: String::new(),
            audio: AudioAvailability::NotKept,
        }
    }

    #[test]
    fn removing_and_retitling_touch_only_the_named_entry() {
        let listing = HistoryListing::new(vec![summary("a", 1), summary("b", 2)], Vec::new());
        let id = HistoryId::parse("a").unwrap();
        let titles = |listing: &HistoryListing| {
            listing
                .summaries
                .iter()
                .map(|summary| summary.title.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(titles(&listing.removed(&id)), vec!["b"]);
        assert_eq!(titles(&listing.retitled(&id, "Notes")), vec!["b", "Notes"]);
        assert_eq!(listing.removed(&HistoryId::parse("zz").unwrap()), listing);
    }

    #[test]
    fn the_newest_entry_comes_first_and_ties_break_by_id() {
        let listing = HistoryListing::new(
            vec![summary("a", 1), summary("c", 5), summary("b", 5)],
            Vec::new(),
        );
        let order: Vec<&str> = listing
            .summaries
            .iter()
            .map(|summary| summary.id.as_str())
            .collect();
        assert_eq!(order, vec!["c", "b", "a"]);
    }
}
