use crate::{
    components::{button::Button, export_menu::ExportSheet, sheet::Sheet},
    history::export_source::export_source,
    platform::{self, Platform as _, PlatformError},
    settings_store::use_settings,
};
use dioxus::prelude::*;
use purr_common::history::entry::{HistoryId, HistoryRecord};

fn describe(error: &PlatformError) -> String {
    if let PlatformError::HistoryEntryNotFound { .. } = error {
        "This transcription is no longer in History.".to_owned()
    } else {
        error.to_string()
    }
}

/// Loads a saved transcription and opens the export sheet on it; nothing is re-run.
#[component]
pub fn ExportHost(id: HistoryId, on_close: EventHandler<()>) -> Element {
    let settings = use_settings();
    let mut format = use_signal(|| settings.peek().ui.export_format);
    let record = use_resource(move || {
        let id = id.clone();
        async move {
            platform::get_platform()
                .get_history_entry(&id)
                .await
                .map_err(|error| describe(&error))
        }
    });
    let loaded: Option<Result<HistoryRecord, String>> = record.read().clone();
    match loaded {
        None => rsx! {},
        Some(Ok(found)) => rsx! {
            ExportSheet {
                source: export_source(&found.entry),
                format: format(),
                on_format: move |next| format.set(next),
                on_close,
            }
        },
        Some(Err(message)) => rsx! {
            Sheet {
                title: "Couldn't open the transcription",
                on_close,
                actions: rsx! {
                    Button { autofocus: true, onclick: move |_| on_close.call(()), "Close" }
                },
                p { role: "alert", "{message}" }
            }
        },
    }
}
