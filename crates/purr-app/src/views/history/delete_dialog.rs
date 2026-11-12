use crate::{
    components::dialog::{Dialog, DialogTone},
    history::store::use_history,
};
use dioxus::prelude::*;
use purr_common::history::entry::HistoryId;

/// Asks before a transcription is deleted; the toast that follows offers Undo.
#[component]
pub fn DeleteDialog(
    id: HistoryId,
    title: String,
    on_close: EventHandler<()>,
    /// Called after the deletion is started, for a page that shows the deleted entry.
    on_deleted: Option<EventHandler<()>>,
) -> Element {
    let store = use_history();
    let name = title.clone();
    rsx! {
        Dialog {
            title: "Delete this transcript?",
            confirm_label: "Delete",
            tone: DialogTone::Destructive,
            on_cancel: on_close,
            on_confirm: move |()| {
                store.delete(id.clone(), title.clone());
                on_close.call(());
                if let Some(handler) = on_deleted {
                    handler.call(());
                }
            },
            "“{name}” will be removed from History."
        }
    }
}
