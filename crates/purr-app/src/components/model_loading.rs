use super::progress_bar::{Progress, ProgressBar};
use crate::utils::byte_size::ByteSize;
use dioxus::prelude::*;

/// Download and initialisation progress of the model a transcription needs.
#[component]
pub fn ModelLoading(loaded_bytes: u64, total_bytes: u64) -> Element {
    let total = total_bytes.max(1);
    let fraction = (loaded_bytes as f64 / total as f64).min(1.0);
    let downloading = loaded_bytes < total_bytes;
    let caption = format!("{} of {}", ByteSize(loaded_bytes), ByteSize(total_bytes));

    rsx! {
        div { class: "dl", role: "status",
            h2 { class: "t3",
                if downloading {
                    "Downloading the model…"
                } else {
                    "Preparing the model…"
                }
            }
            if downloading {
                ProgressBar {
                    progress: Progress::Determinate(fraction),
                    label: "Model download",
                    value_text: caption.clone(),
                }
                div { class: "dlmeta",
                    span { "{caption}" }
                    span { "Saved on this device, so this only happens once." }
                }
            } else {
                ProgressBar { progress: Progress::Indeterminate, label: "Loading the model" }
                p { class: "foot", style: "margin:0", "Loading the model onto the GPU. Large models can take a minute." }
            }
        }
    }
}
