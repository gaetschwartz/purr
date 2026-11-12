//! The Transcribe screen: the chosen file, the run and the result, in place on `/`.

mod cancelled_notice;
mod engine_notice;
mod failed;
mod model_notice;
mod recent;
mod result;
mod running;
mod start;

use crate::transcribe::{run::RunOutcome, session::use_transcribe_session};
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
    match record.run.outcome.clone() {
        RunOutcome::Running => rsx! {
            running::Running { record }
        },
        RunOutcome::Completed(completion) => rsx! {
            result::ResultView { record, completion }
        },
        RunOutcome::Failed(failure) => rsx! {
            failed::Failed { record, failure }
        },
    }
}
