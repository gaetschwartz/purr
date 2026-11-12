use crate::{
    components::dialog::{Dialog, DialogTone},
    history::{store::use_history, wording::transcriptions},
};
use dioxus::prelude::*;

/// Asks before every saved transcription is deleted.
#[component]
pub fn ClearDialog(count: usize, unreadable: usize, on_close: EventHandler<()>) -> Element {
    let store = use_history();
    let extra = if unreadable == 0 {
        String::new()
    } else {
        format!(" and {unreadable} that can't be read")
    };
    rsx! {
        Dialog {
            title: "Clear all history?",
            confirm_label: "Clear all history",
            tone: DialogTone::Destructive,
            on_cancel: on_close,
            on_confirm: move |()| {
                store.clear();
                on_close.call(());
            },
            "This deletes {transcriptions(count)}{extra}. This can't be undone."
        }
    }
}
