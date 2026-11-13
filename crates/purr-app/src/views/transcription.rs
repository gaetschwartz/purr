//! `/transcription/:id`: a transcription's address, which exists from the moment it starts.

mod engine_notice;
mod run_view;
mod saved;

use crate::transcribe::session::use_transcribe_session;
use dioxus::prelude::*;
use purr_common::history::entry::HistoryId;
use run_view::RunView;
use saved::SavedTranscription;

/// The live run while the session holds the transcription `id` as its run, else the saved
/// transcription of that id. The run keeps its address once it is saved, so nothing remounts.
#[component]
pub fn Transcription(id: HistoryId) -> Element {
    let session = use_transcribe_session();
    match session.record().filter(|record| record.id == id) {
        Some(record) => rsx! {
            RunView { key: "{record.id}", record }
        },
        None => rsx! {
            SavedTranscription { key: "{id}", id }
        },
    }
}
