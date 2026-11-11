//! What exporting a saved transcription describes.

use crate::components::export_menu::ExportSource;
use purr_common::history::entry::HistoryEntry;
use std::rc::Rc;

/// The export of `entry`, named after its audio file and made of the stored segments.
#[must_use]
pub fn export_source(entry: &HistoryEntry) -> ExportSource {
    ExportSource {
        file_name: entry.file_name.clone(),
        model: entry.model_id.clone(),
        language: entry.language.clone(),
        duration_seconds: entry.audio_seconds,
        segments: Rc::new(entry.segments.clone()),
    }
}
