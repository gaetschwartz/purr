//! The Transcribe screen: the chosen file, the run and the result, in place on `/`.

mod cancelled_notice;
mod engine_notice;
mod model_notice;
mod recent;
mod run_view;
mod start;

use crate::transcribe::session::use_transcribe_session;
use dioxus::prelude::*;

/// The Transcribe view. The run lives in the root session, so leaving and coming back keeps it.
#[component]
pub fn Home() -> Element {
    let session = use_transcribe_session();

    let Some(record) = session.record() else {
        return rsx! {
            start::Start {}
        };
    };
    rsx! {
        run_view::RunView {
            key: "{record.file.id}-{record.run.started_ms()}",
            record,
        }
    }
}
