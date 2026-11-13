//! How much audio History keeps in the browser: a share of the storage the browser grants the
//! site, with the oldest audio dropped first.

use super::entry::HistoryId;

/// The share of the browser's storage quota that History audio may use, in percent.
pub const BUDGET_PERCENT: u64 = 50;

/// The bytes of audio History may keep under a storage `quota`.
#[must_use]
pub fn budget(quota: u64) -> u64 {
    let share = u128::from(quota) * u128::from(BUDGET_PERCENT) / 100;
    u64::try_from(share).unwrap_or(u64::MAX)
}

/// A stored audio file, as the budget sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredAudio {
    pub id: HistoryId,
    pub created_at_ms: i64,
    pub bytes: u64,
}

/// What to do to store one more audio file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioPlan {
    /// Store it after dropping the audio of these entries, oldest first.
    Store { drop_first: Vec<HistoryId> },
    /// It is larger than the whole budget: store nothing and drop nothing.
    TooLarge,
}

/// The plan to store `incoming` bytes for the entry `replacing`, whose own stored audio (if any)
/// is replaced and so does not count.
#[must_use]
pub fn plan(
    stored: &[StoredAudio],
    replacing: &HistoryId,
    incoming: u64,
    budget: u64,
) -> AudioPlan {
    if incoming > budget {
        return AudioPlan::TooLarge;
    }
    let mut others: Vec<&StoredAudio> = stored
        .iter()
        .filter(|audio| &audio.id != replacing)
        .collect();
    others.sort_by_key(|audio| (audio.created_at_ms, &audio.id));
    let mut total = others
        .iter()
        .fold(incoming, |sum, audio| sum.saturating_add(audio.bytes));
    let mut drop_first = Vec::new();
    for audio in others {
        if total <= budget {
            break;
        }
        total = total.saturating_sub(audio.bytes);
        drop_first.push(audio.id.clone());
    }
    AudioPlan::Store { drop_first }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(name: &str) -> HistoryId {
        HistoryId::parse(name).unwrap_or_else(|_| unreachable!())
    }

    fn stored(name: &str, created_at_ms: i64, bytes: u64) -> StoredAudio {
        StoredAudio {
            id: id(name),
            created_at_ms,
            bytes,
        }
    }

    #[test]
    fn the_budget_is_half_of_the_quota() {
        assert_eq!(budget(1_000), 500);
        assert_eq!(budget(0), 0);
        assert_eq!(budget(u64::MAX), u64::MAX / 2);
    }

    #[test]
    fn nothing_is_dropped_while_everything_fits() {
        let audio = [stored("a", 1, 40), stored("b", 2, 40)];
        assert_eq!(
            plan(&audio, &id("c"), 20, 100),
            AudioPlan::Store {
                drop_first: Vec::new()
            }
        );
    }

    #[test]
    fn the_oldest_audio_goes_first_until_the_new_file_fits() {
        let audio = [stored("b", 2, 40), stored("a", 1, 40), stored("c", 3, 15)];
        assert_eq!(
            plan(&audio, &id("d"), 50, 100),
            AudioPlan::Store {
                drop_first: vec![id("a"), id("b")]
            }
        );
    }

    #[test]
    fn a_resaved_entry_does_not_count_against_itself() {
        let audio = [stored("a", 1, 60), stored("b", 2, 30)];
        assert_eq!(
            plan(&audio, &id("a"), 60, 100),
            AudioPlan::Store {
                drop_first: Vec::new()
            }
        );
    }

    #[test]
    fn a_file_larger_than_the_budget_changes_nothing() {
        let audio = [stored("a", 1, 10)];
        assert_eq!(plan(&audio, &id("b"), 101, 100), AudioPlan::TooLarge);
    }

    #[test]
    fn equal_times_drop_by_id() {
        let audio = [stored("b", 5, 60), stored("a", 5, 60)];
        assert_eq!(
            plan(&audio, &id("c"), 30, 100),
            AudioPlan::Store {
                drop_first: vec![id("a")]
            }
        );
    }
}
