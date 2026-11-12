//! Turns a finished run into the entry that is saved to History.

use purr_common::{
    history::{
        entry::{
            HistoryEntry, HistoryId, HistorySource, InvalidHistoryId, LanguageOrigin,
            quantize_peaks,
        },
        preview::preview,
        title::default_title,
    },
    platform::TranscriptSegment,
};
use std::path::PathBuf;
use uuid::Uuid;

/// What a finished run knows.
#[derive(Debug, Clone)]
pub struct Finished {
    pub id: HistoryId,
    pub created_at_ms: i64,
    pub file_name: String,
    /// The original file on the computer, when the platform says where it is.
    pub source_path: Option<PathBuf>,
    pub model_id: String,
    pub model_label: String,
    pub language: Option<String>,
    pub language_origin: LanguageOrigin,
    pub processing_seconds: f64,
    pub audio_seconds: f64,
    pub segments: Vec<TranscriptSegment>,
    /// Waveform bars between 0 and 1, when the decoder reported them.
    pub peaks: Vec<f32>,
}

/// A fresh id for an entry.
pub fn new_id() -> Result<HistoryId, InvalidHistoryId> {
    HistoryId::parse(&Uuid::new_v4().to_string())
}

/// The entry for `run`: titled after the file, with a preview of what was said.
#[must_use]
pub fn build(run: Finished) -> HistoryEntry {
    HistoryEntry {
        id: run.id,
        title: default_title(&run.file_name),
        preview: preview(&run.segments),
        source: run
            .source_path
            .map_or(HistorySource::NotKept, |path| HistorySource::Path { path }),
        peaks: quantize_peaks(&run.peaks),
        file_name: run.file_name,
        created_at_ms: run.created_at_ms,
        audio_seconds: run.audio_seconds,
        language: run.language,
        language_origin: run.language_origin,
        model_id: run.model_id,
        model_label: run.model_label,
        processing_seconds: run.processing_seconds,
        segments: run.segments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finished() -> Finished {
        Finished {
            id: HistoryId::parse("run-1").unwrap(),
            created_at_ms: 1_762_848_000_000,
            file_name: "WhatsApp Ptt.ogg".to_owned(),
            source_path: Some(PathBuf::from("C:/audio/WhatsApp Ptt.ogg")),
            model_id: "tiny".to_owned(),
            model_label: "Tiny".to_owned(),
            language: Some("fr".to_owned()),
            language_origin: LanguageOrigin::Detected,
            processing_seconds: 3.0,
            audio_seconds: 12.0,
            segments: vec![
                TranscriptSegment {
                    text: " Bonjour à tous.".to_owned(),
                    start_time: 0.0,
                    end_time: 2.0,
                    words: Vec::new(),
                },
                TranscriptSegment {
                    text: " Merci d'être là.".to_owned(),
                    start_time: 2.0,
                    end_time: 4.0,
                    words: Vec::new(),
                },
            ],
            peaks: vec![0.0, 0.5, 1.0],
        }
    }

    #[test]
    fn a_new_entry_is_titled_after_the_file_and_previews_the_speech() {
        let entry = build(finished());
        assert_eq!(entry.title, "WhatsApp Ptt.ogg");
        assert_eq!(entry.preview, "Bonjour à tous. Merci d'être là.");
        assert_eq!(entry.segments.len(), 2);
    }

    #[test]
    fn the_original_path_is_kept_only_when_known() {
        assert_eq!(
            build(finished()).source,
            HistorySource::Path {
                path: PathBuf::from("C:/audio/WhatsApp Ptt.ogg")
            }
        );
        let browser = Finished {
            source_path: None,
            ..finished()
        };
        assert_eq!(build(browser).source, HistorySource::NotKept);
    }

    #[test]
    fn peaks_are_stored_as_levels() {
        assert_eq!(build(finished()).peaks, vec![0, 128, 255]);
        let silent = Finished {
            peaks: Vec::new(),
            ..finished()
        };
        assert_eq!(build(silent).peaks, Vec::<u8>::new());
    }

    #[test]
    fn a_run_without_speech_has_an_empty_preview() {
        let quiet = Finished {
            segments: Vec::new(),
            ..finished()
        };
        assert_eq!(build(quiet).preview, "");
    }

    #[test]
    fn fresh_ids_differ_and_are_valid() {
        let first = new_id().unwrap();
        assert_ne!(first, new_id().unwrap());
    }
}
