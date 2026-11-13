use super::{
    icon::{Icon, IconName, IconSize},
    spinner::Spinner,
};
use dioxus::{html::FileData, html::HasFileData, prelude::*};

/// The DOM id of the pad's file input; open the chooser by clicking it.
pub const FILE_INPUT_ID: &str = "file-input";

/// The drop area of the Transcribe screen: drop a file on it, or click it (or press Space or
/// Enter on it) to open the file chooser. It "purrs" a few times when it appears, unless motion is reduced.
#[component]
pub fn DropPad(
    /// The name of the file being read, while one is.
    reading: Option<String>,
    /// The size limit of a file, when the platform has one.
    limit: Option<String>,
    on_file: EventHandler<FileData>,
) -> Element {
    let mut over = use_signal(|| false);
    let busy = reading.is_some();
    let class = if over() {
        "drop surface over"
    } else {
        "drop surface idle"
    };

    rsx! {
        label {
            class,
            title: limit.clone(),
            ondragenter: move |event: DragEvent| {
                event.prevent_default();
                over.set(true);
            },
            ondragover: move |event: DragEvent| {
                event.prevent_default();
                over.set(true);
            },
            ondragleave: move |_| over.set(false),
            ondrop: move |event: DragEvent| {
                event.prevent_default();
                over.set(false);
                if let Some(file) = event.data().files().into_iter().next() {
                    on_file.call(file);
                }
            },
            input {
                r#type: "file",
                class: "sr",
                id: FILE_INPUT_ID,
                name: "audio-file",
                multiple: false,
                disabled: busy,
                aria_label: "Choose an audio file",
                aria_describedby: limit.is_some().then_some("drop-limit"),
                onchange: move |event: FormEvent| {
                    if let Some(file) = event.files().into_iter().next() {
                        on_file.call(file);
                    }
                },
            }
            if let Some(name) = reading {
                span { class: "halo",
                    Spinner { large: true }
                }
                h2 { role: "status", "Reading the file…" }
                p { "{name}" }
            } else {
                span { class: "halo",
                    Icon { name: IconName::Upload, size: IconSize::Large }
                }
                h2 { "Drop an audio file" }
                p {
                    "or "
                    span { class: "link", "choose a file" }
                }
                if let Some(limit) = &limit {
                    span { class: "sr", id: "drop-limit", "{limit}" }
                }
            }
        }
    }
}
