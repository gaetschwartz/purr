//! The audio of a saved transcription when the platform stores it itself, and what saving it did.

use bytes::Bytes;

/// An audio file as the platform keeps it for a saved transcription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryAudio {
    pub bytes: Bytes,
    pub content_type: Option<String>,
}

/// Why a saved transcription ended up without the audio that was offered with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioDropReason {
    /// The file alone is larger than the share of storage reserved for History audio.
    OverBudget,
    /// The browser refused the write because the site is out of storage.
    StorageFull,
    Failed {
        detail: String,
    },
}

/// What saving an entry did with the audio offered with it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AudioOutcome {
    /// No audio was offered, or the platform keeps the original file instead.
    #[default]
    NotOffered,
    Kept,
    /// The transcript is saved, without its audio.
    Dropped(AudioDropReason),
}
