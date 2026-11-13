//! The stored form of an entry: JSON in a versioned envelope, identical on every platform.

use super::entry::HistoryEntry;
use serde::{Deserialize, Serialize};

/// The version this build writes. Version 2 added the words of each segment.
pub const FORMAT_VERSION: u32 = 2;

/// The oldest version this build still reads; its entries read as having no words.
const OLDEST_READABLE: u32 = 1;

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
        OLDEST_READABLE..=FORMAT_VERSION => {
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
        accel::{AccelBackend, EngineUsed},
        history::entry::{HistoryId, HistorySource, LanguageOrigin},
        platform::TranscriptSegment,
        word::Word,
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
                words: vec![Word {
                    range: 0..7,
                    start_cs: Some(10),
                    end_cs: Some(140),
                    score: Some(230),
                }],
            }],
            peaks: vec![0, 128, 255],
            source: HistorySource::Path {
                path: PathBuf::from("memo.ogg"),
            },
            engine: Some(EngineUsed {
                backend: AccelBackend::Vulkan,
                hardware: Some("Test GPU".to_owned()),
            }),
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
            Some("{\"version\":2")
        );
    }

    #[test]
    fn words_are_stored_compactly() {
        let stored = encode(&entry()).unwrap_or_default();
        assert_eq!(
            stored
                .split_once("\"words\":")
                .and_then(|(_, rest)| rest.split_once('}'))
                .map(|(words, _)| words),
            Some("[[0,7,10,140,230]]")
        );
    }

    #[test]
    fn a_version_1_entry_reads_without_words() {
        let stored = "{\"version\":1,\"entry\":{\"id\":\"abc-1\",\"title\":\"memo.ogg\",\"file_name\":\"memo.ogg\",\"created_at_ms\":1762848000000,\"audio_seconds\":12.5,\"language\":\"fr\",\"language_origin\":\"detected\",\"model_id\":\"tiny\",\"model_label\":\"Tiny\",\"processing_seconds\":2.0,\"preview\":\"Bonjour\",\"segments\":[{\"text\":\"Bonjour\",\"start_time\":0.0,\"end_time\":1.5}],\"peaks\":[0,128,255],\"source\":{\"kind\":\"path\",\"path\":\"memo.ogg\"}}}";
        let mut expected = entry();
        expected.segments[0].words.clear();
        expected.engine = None;
        assert_eq!(decode(stored), Ok(expected));
    }

    #[test]
    fn a_version_1_entry_is_saved_again_as_version_2() {
        let mut read = entry();
        read.segments[0].words.clear();
        let stored = encode(&read).unwrap_or_default();
        assert_eq!(
            stored.split_once(',').map(|(head, _)| head),
            Some("{\"version\":2")
        );
        assert_eq!(decode(&stored), Ok(read));
    }

    #[test]
    fn a_newer_version_is_named_not_guessed_at() {
        assert_eq!(
            decode("{\"version\":9,\"entry\":{\"future\":true}}"),
            Err(UnreadableReason::NewerVersion { found: 9 })
        );
        assert_eq!(
            decode("{\"version\":3,\"entry\":{}}"),
            Err(UnreadableReason::NewerVersion { found: 3 })
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
