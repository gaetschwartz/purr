//! Whether the result on screen has been saved to History.

use purr_common::history::entry::{HistoryEntry, HistoryId};
use std::rc::Rc;

/// Where saving the last finished run stands.
#[derive(Debug, Clone, PartialEq)]
pub enum SaveState {
    /// Nothing to save: no finished run yet.
    Idle,
    Saving,
    Saved(HistoryId),
    /// The save failed; the entry is kept so Retry can write it again.
    Failed {
        message: String,
        entry: Rc<HistoryEntry>,
    },
}
