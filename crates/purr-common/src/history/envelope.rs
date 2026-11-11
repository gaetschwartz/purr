//! The stored form of an entry: JSON in a versioned envelope, identical on every platform.

use super::entry::HistoryEntry;
use serde::{Deserialize, Serialize};

/// The version this build writes and reads.
pub const FORMAT_VERSION: u32 = 1;

/// Why a stored entry could not be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UnreadableReason {
    #[error("it is damaged ({detail})")]
    Corrupt { detail: String },
    #[error("it was saved by a newer version of Purr (format {found})")]
    NewerVersion { found: u32 },
}

#[derive(Serialize)]
struct Writing<'a> {
    version: u32,
    entry: &'a HistoryEntry,
}

#[derive(Deserialize)]
struct Head {
    version: u32,
}

#[derive(Deserialize)]
struct Reading {
    entry: HistoryEntry,
}

/// The text to store for `entry`.
pub fn encode(entry: &HistoryEntry) -> serde_json::Result<String> {
    serde_json::to_string(&Writing {
        version: FORMAT_VERSION,
        entry,
    })
}

/// The entry stored as `text`.
pub fn decode(text: &str) -> Result<HistoryEntry, UnreadableReason> {
    let corrupt = |error: serde_json::Error| UnreadableReason::Corrupt {
        detail: error.to_string(),
    };
    let head: Head = serde_json::from_str(text).map_err(corrupt)?;
    match head.version {
        FORMAT_VERSION => {
            let reading: Reading = serde_json::from_str(text).map_err(corrupt)?;
            Ok(reading.entry)
        }
        0 => Err(UnreadableReason::Corrupt {
            detail: "version 0 never existed".to_owned(),
        }),
        found => Err(UnreadableReason::NewerVersion { found }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        history::entry::{HistoryId, HistorySource, LanguageOrigin},
        platform::TranscriptSegment,
    };
    use std::path::PathBuf;

    fn entry() -> HistoryEntry {
        HistoryEntry {
            id: HistoryId::parse("abc-1").unwrap_or_else(|_| unreachable!()),
            title: "memo.ogg".to_owned(),
            file_name: "memo.ogg".to_owned(),
            created_at_ms: 1_762_848_000_000,
            audio_seconds: 12.5,
            language: Some("fr".to_owned()),
            language_origin: LanguageOrigin::Detected,
            model_id: "tiny".to_owned(),
            model_label: "Tiny".to_owned(),
            processing_seconds: 2.0,
            preview: "Bonjour".to_owned(),
            segments: vec![TranscriptSegment {
                text: "Bonjour".to_owned(),
                start_time: 0.0,
                end_time: 1.5,
            }],
            peaks: vec![0, 128, 255],
            source: HistorySource::Path {
                path: PathBuf::from("memo.ogg"),
            },
        }
    }

    #[test]
    fn an_entry_survives_the_round_trip() {
        let stored = encode(&entry()).unwrap();
        assert_eq!(decode(&stored), Ok(entry()));
    }

    #[test]
    fn the_envelope_carries_the_version() {
        let stored = encode(&entry()).unwrap_or_default();
        assert_eq!(
            stored.split_once(',').map(|(head, _)| head),
            Some("{\"version\":1")
        );
    }

    #[test]
    fn a_newer_version_is_named_not_guessed_at() {
        assert_eq!(
            decode("{\"version\":9,\"entry\":{\"future\":true}}"),
            Err(UnreadableReason::NewerVersion { found: 9 })
        );
    }

    #[test]
    fn garbage_is_corrupt() {
        assert!(matches!(
            decode("not json at all"),
            Err(UnreadableReason::Corrupt { .. })
        ));
        assert!(matches!(
            decode("{\"version\":1,\"entry\":{\"id\":\"abc\"}}"),
            Err(UnreadableReason::Corrupt { .. })
        ));
        assert!(matches!(
            decode("{\"version\":0}"),
            Err(UnreadableReason::Corrupt { .. })
        ));
    }

    #[test]
    fn an_id_that_is_not_a_file_name_is_corrupt() {
        let stored = encode(&entry())
            .unwrap_or_default()
            .replace("abc-1", "../x");
        assert!(matches!(
            decode(&stored),
            Err(UnreadableReason::Corrupt { .. })
        ));
    }
}
