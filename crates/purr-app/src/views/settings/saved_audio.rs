use crate::{
    components::{
        button::{Button, ButtonSize},
        dialog::{Dialog, DialogTone},
        group::{Group, Item},
        presence::Presence,
    },
    history::store::use_history,
    platform::{self, Platform as _},
    utils::byte_size::ByteSize,
};
use dioxus::prelude::*;

/// The audio History keeps in this browser, with a way to delete all of it; web only.
#[component]
pub fn SavedAudioGroup() -> Element {
    let store = use_history();
    let mut confirming = use_signal(|| false);
    let usage = use_resource(move || async move {
        let _reloaded_with = store.load();
        platform::get_platform()
            .history_audio_bytes()
            .await
            .map_err(|error| {
                tracing::error!("Could not measure the saved audio: {error}");
                error.to_string()
            })
    });
    let used: Option<Result<u64, String>> = usage.read().clone();
    let (size, empty) = match &used {
        None => ("…".to_owned(), true),
        Some(Ok(bytes)) => (ByteSize(*bytes).to_string(), *bytes == 0),
        Some(Err(_)) => ("Unknown".to_owned(), false),
    };

    rsx! {
        Group {
            Item {
                title: "Saved audio",
                subtitle: "History keeps the audio of your transcriptions in this browser, so you can play them back.",
                value: size,
                control: rsx! {
                    Button {
                        size: ButtonSize::Small,
                        disabled: empty,
                        onclick: move |_| confirming.set(true),
                        "Clear"
                    }
                },
            }
        }
        Presence { present: confirming(),
            Dialog {
                title: "Clear saved audio?",
                confirm_label: "Clear audio",
                tone: DialogTone::Destructive,
                on_confirm: move |()| {
                    confirming.set(false);
                    store.clear_audio();
                },
                on_cancel: move |()| confirming.set(false),
                "Transcriptions stay in History, but their audio can no longer be played back."
            }
        }
    }
}
