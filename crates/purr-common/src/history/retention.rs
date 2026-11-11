//! How much history is kept: the newest entries within a count and a size, the rest dropped.

use super::entry::HistoryId;
use std::cmp::Reverse;

/// The most entries kept.
pub const MAX_ENTRIES: usize = 200;

/// The most bytes of stored entries kept (JSON text, as written).
pub const MAX_BYTES: u64 = 32 * 1024 * 1024;

/// What is kept at most.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub max_entries: usize,
    pub max_bytes: u64,
}

impl Limits {
    /// The limits of the app.
    pub const STANDARD: Self = Self {
        max_entries: MAX_ENTRIES,
        max_bytes: MAX_BYTES,
    };
}

/// An entry as storage knows it, without reading its content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stored {
    pub id: HistoryId,
    pub created_at_ms: i64,
    pub bytes: u64,
}

/// The ids to delete so that what is left fits `limits`, oldest first.
///
/// Entries are ranked newest first. The newest one is always kept, even when it alone is larger
/// than `limits.max_bytes`. After the first entry that does not fit, every older one goes too, so
/// a small old entry never outlives a bigger newer one.
#[must_use]
pub fn evictions(stored: &[Stored], limits: Limits) -> Vec<HistoryId> {
    let mut ranked: Vec<&Stored> = stored.iter().collect();
    ranked.sort_by_key(|entry| Reverse((entry.created_at_ms, &entry.id)));
    let mut kept = 0_usize;
    let mut kept_bytes = 0_u64;
    let mut full = false;
    let mut dropped = Vec::new();
    for entry in ranked {
        let fits = kept == 0
            || (kept < limits.max_entries
                && kept_bytes.saturating_add(entry.bytes) <= limits.max_bytes);
        if fits && !full {
            kept += 1;
            kept_bytes = kept_bytes.saturating_add(entry.bytes);
        } else {
            full = true;
            dropped.push(entry.id.clone());
        }
    }
    dropped.reverse();
    dropped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored(name: &str, created_at_ms: i64, bytes: u64) -> Stored {
        Stored {
            id: HistoryId::parse(name).unwrap_or_else(|_| unreachable!()),
            created_at_ms,
            bytes,
        }
    }

    fn names(ids: &[HistoryId]) -> Vec<&str> {
        ids.iter().map(HistoryId::as_str).collect()
    }

    const SMALL: Limits = Limits {
        max_entries: 3,
        max_bytes: 100,
    };

    #[test]
    fn nothing_is_dropped_within_the_limits() {
        let entries = [stored("a", 1, 10), stored("b", 2, 10), stored("c", 3, 10)];
        assert_eq!(evictions(&entries, SMALL), Vec::new());
    }

    #[test]
    fn the_count_cap_drops_the_oldest() {
        let entries = [
            stored("d", 4, 1),
            stored("a", 1, 1),
            stored("c", 3, 1),
            stored("b", 2, 1),
        ];
        assert_eq!(names(&evictions(&entries, SMALL)), vec!["a"]);
    }

    #[test]
    fn the_size_cap_drops_the_oldest_until_it_fits() {
        let entries = [stored("a", 1, 40), stored("b", 2, 40), stored("c", 3, 40)];
        assert_eq!(names(&evictions(&entries, SMALL)), vec!["a"]);
    }

    #[test]
    fn a_small_old_entry_does_not_outlive_a_big_newer_one() {
        let entries = [stored("a", 1, 5), stored("b", 2, 90), stored("c", 3, 20)];
        assert_eq!(names(&evictions(&entries, SMALL)), vec!["a", "b"]);
    }

    #[test]
    fn the_newest_entry_is_kept_even_when_it_exceeds_the_size_cap() {
        let entries = [stored("a", 1, 10), stored("b", 2, 500)];
        assert_eq!(names(&evictions(&entries, SMALL)), vec!["a"]);
    }

    #[test]
    fn equal_times_are_ranked_by_id() {
        let entries = [
            stored("a", 5, 1),
            stored("b", 5, 1),
            stored("c", 5, 1),
            stored("d", 5, 1),
        ];
        assert_eq!(names(&evictions(&entries, SMALL)), vec!["a"]);
    }

    #[test]
    fn the_standard_limits_hold_two_hundred_entries() {
        let entries: Vec<Stored> = (0..250)
            .map(|index| stored(&format!("e{index:03}"), i64::from(index), 1_000))
            .collect();
        let dropped = evictions(&entries, Limits::STANDARD);
        assert_eq!(dropped.len(), 50);
        assert_eq!(names(&dropped).first(), Some(&"e000"));
        assert_eq!(names(&dropped).last(), Some(&"e049"));
    }
}
