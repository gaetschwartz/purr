use super::{
    icon::{Icon, IconName, IconSize},
    spinner::Spinner,
};
use dioxus::{html::FileData, html::HasFileData, prelude::*};

/// The DOM id of the pad's file input; open the chooser by clicking it.
pub const FILE_INPUT_ID: &str = "file-input";

/// What the file chooser offers.
const ACCEPT: &str = "audio/*,.mp3,.wav,.aac,.flac,.ogg,.oga,.opus,.webm,.weba,.m4a,.m4b,.mp4,.wma,.aif,.aiff,.amr,.mka";

/// The glass well of the Transcribe screen: drop a file on it, or click it (or press Space or
/// Enter on it) to open the file chooser. It "purrs" gently while idle, unless motion is reduced.
#[component]
pub fn DropPad(
    /// The name of the file being read, while one is.
    reading: Option<String>,
    /// The formats line under the title.
    #[props(into)]
    formats: String,
    /// The Cmd or Ctrl label for the shortcut hint.
    #[props(into)]
    primary_key: String,
    on_file: EventHandler<FileData>,
) -> Element {
    let mut over = use_signal(|| false);
    let busy = reading.is_some();
    let class = if over() {
        "drop glass over"
    } else {
        "drop glass idle"
    };

    rsx! {
        label {
            class,
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
            span { class: "well" }
            input {
                r#type: "file",
                class: "sr",
                id: FILE_INPUT_ID,
                name: "audio-file",
                multiple: false,
                disabled: busy,
                aria_label: "Choose an audio file",
                aria_describedby: "drop-formats",
                accept: ACCEPT,
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
                    " · {primary_key}+O"
                }
                span { class: "formats", id: "drop-formats", "{formats}" }
            }
        }
    }
}
