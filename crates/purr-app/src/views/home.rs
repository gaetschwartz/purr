//! The Transcribe screen: first run, the chosen file, the run and the result, in place on `/`.

mod failed;
mod first_run;
mod model_notice;
mod result;
mod running;
mod start;

use crate::{
    components::{page::Page, spinner::Spinner},
    hooks::use_models::use_models,
    transcribe::{run::RunOutcome, session::use_transcribe_session},
};
use dioxus::prelude::*;

/// The Transcribe view. The run lives in the root session, so leaving and coming back keeps it.
#[component]
pub fn Home() -> Element {
    let session = use_transcribe_session();
    let models = use_models();

    if let Some(record) = session.record() {
        return match record.run.outcome.clone() {
            RunOutcome::Running => rsx! {
                running::Running { record }
            },
            RunOutcome::Completed(completion) => rsx! {
                result::ResultView { record, completion }
            },
            RunOutcome::Failed(failure) => rsx! {
                failed::Failed { record, failure }
            },
        };
    }

    let any_on_device = models
        .rows()
        .map(|rows| rows.map(|rows| rows.iter().any(|row| row.state.is_on_device())));
    match any_on_device {
        None => rsx! {
            Page { title: "Transcribe",
                div { class: "row", role: "status",
                    Spinner {}
                    span { class: "muted", "Looking for models on this device…" }
                }
            }
        },
        Some(Ok(false)) => rsx! {
            first_run::FirstRun {}
        },
        Some(Ok(true) | Err(_)) => rsx! {
            start::Start {}
        },
    }
}
