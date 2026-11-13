//! A saved transcription, and the lighter views of it the lists use.

use super::preview::search_text;
use crate::{accel::EngineUsed, platform::TranscriptSegment};
use serde::{Deserialize, Serialize};
use std::{fmt, path::PathBuf, str::FromStr};

const MAX_ID_LEN: usize = 64;

/// A history id was empty, too long, or held a character a file name should not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a history id is 1 to 64 letters, digits or hyphens")]
pub struct InvalidHistoryId;

/// Identifies a saved transcription; safe to use as a file name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct HistoryId(String);

impl HistoryId {
    /// The id spelled `id`, when it is a valid one.
    pub fn parse(id: &str) -> Result<Self, InvalidHistoryId> {
        let valid = (1..=MAX_ID_LEN).contains(&id.len())
            && id
                .chars()
                .all(|letter| letter.is_ascii_alphanumeric() || letter == '-');
        if valid {
            Ok(Self(id.to_owned()))
        } else {
            Err(InvalidHistoryId)
        }
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for HistoryId {
    type Error = InvalidHistoryId;

    fn try_from(id: String) -> Result<Self, Self::Error> {
        Self::parse(&id)
    }
}

impl From<HistoryId> for String {
    fn from(id: HistoryId) -> Self {
        id.0
    }
}

impl FromStr for HistoryId {
    type Err = InvalidHistoryId;

    fn from_str(id: &str) -> Result<Self, Self::Err> {
        Self::parse(id)
    }
}

impl fmt::Display for HistoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Where the audio of a saved transcription can be found again, as it was saved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HistorySource {
    /// The audio was not kept: the browser does not copy it into history.
    NotKept,
    /// The original file on the computer.
    Path { path: PathBuf },
}

/// Whether the audio of a saved transcription can be played now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioAvailability {
    Available,
    /// The original file moved or was deleted.
    Missing,
    NotKept,
}

/// How the spoken language of a transcription was settled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LanguageOrigin {
    /// The model recognised it.
    Detected,
    /// The user asked for it.
    Chosen,
}

/// A saved transcription, as it is written to storage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: HistoryId,
    /// What the list shows; renamable, starts as the file name.
    pub title: String,
    /// The audio file's own name.
    pub file_name: String,
    /// Milliseconds since the Unix epoch.
    pub created_at_ms: i64,
    pub audio_seconds: f64,
    /// The Whisper code of the spoken language, when known.
    pub language: Option<String>,
    pub language_origin: LanguageOrigin,
    pub model_id: String,
    pub model_label: String,
    pub processing_seconds: f64,
    /// The first words of the transcript.
    pub preview: String,
    pub segments: Vec<TranscriptSegment>,
    /// Waveform bars, each from 0 (silent) to 255 (the loudest bar).
    pub peaks: Vec<u8>,
    pub source: HistorySource,
    /// What ran the transcription; absent for entries saved before it was recorded and in the
    /// browser.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<EngineUsed>,
}

impl HistoryEntry {
    /// The waveform bars between 0 and 1.
    #[must_use]
    pub fn waveform(&self) -> Vec<f32> {
        self.peaks
            .iter()
            .map(|level| f32::from(*level) / 255.0)
            .collect()
    }

    /// The light view of the entry for lists.
    #[must_use]
    pub fn summary(&self, audio: AudioAvailability) -> HistorySummary {
        HistorySummary {
            id: self.id.clone(),
            title: self.title.clone(),
            file_name: self.file_name.clone(),
            created_at_ms: self.created_at_ms,
            audio_seconds: self.audio_seconds,
            language: self.language.clone(),
            language_origin: self.language_origin,
            model_id: self.model_id.clone(),
            model_label: self.model_label.clone(),
            processing_seconds: self.processing_seconds,
            preview: self.preview.clone(),
            segment_count: self.segments.len(),
            search_text: search_text(&self.segments),
            audio,
        }
    }
}

/// Waveform bars from 0 to 1 as the stored 0 to 255 levels.
#[must_use]
pub fn quantize_peaks(bars: &[f32]) -> Vec<u8> {
    bars.iter()
        .map(|bar| (bar.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

/// What a history list needs of an entry, without its segments.
#[derive(Debug, Clone, PartialEq)]
pub struct HistorySummary {
    pub id: HistoryId,
    pub title: String,
    pub file_name: String,
    pub created_at_ms: i64,
    pub audio_seconds: f64,
    pub language: Option<String>,
    pub language_origin: LanguageOrigin,
    pub model_id: String,
    pub model_label: String,
    pub processing_seconds: f64,
    pub preview: String,
    pub segment_count: usize,
    /// The whole transcript in lower case, for searching.
    pub search_text: String,
    pub audio: AudioAvailability,
}

/// A saved transcription with what is known about its audio right now.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryRecord {
    pub entry: HistoryEntry,
    pub audio: AudioAvailability,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_safe_file_names() {
        assert!(HistoryId::parse("0d6c-41AF").is_ok());
        assert_eq!(HistoryId::parse(""), Err(InvalidHistoryId));
        assert_eq!(HistoryId::parse("../evil"), Err(InvalidHistoryId));
        assert_eq!(HistoryId::parse("a b"), Err(InvalidHistoryId));
        assert_eq!(HistoryId::parse(&"a".repeat(65)), Err(InvalidHistoryId));
    }

    #[test]
    fn an_invalid_id_does_not_deserialize() {
        assert!(serde_json::from_str::<HistoryId>("\"../x\"").is_err());
        assert_eq!(
            serde_json::from_str::<HistoryId>("\"ok-1\"").ok(),
            HistoryId::parse("ok-1").ok()
        );
    }

    #[test]
    fn peaks_round_trip_within_one_level() {
        let bars = [0.0, 0.25, 0.5, 1.0];
        let levels = quantize_peaks(&bars);
        assert_eq!(levels, vec![0, 64, 128, 255]);
        let entry_bars: Vec<f32> = levels
            .iter()
            .map(|level| f32::from(*level) / 255.0)
            .collect();
        for (original, restored) in bars.iter().zip(entry_bars) {
            assert!((original - restored).abs() < 1.0 / 255.0);
        }
    }

    #[test]
    fn out_of_range_bars_are_clamped() {
        assert_eq!(quantize_peaks(&[-1.0, 2.0]), vec![0, 255]);
    }
}
