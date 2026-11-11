use super::ProgressBar;
use crate::utils::byte_size::ByteSize;
use dioxus::prelude::*;

/// Download and initialisation progress of the model a transcription needs.
#[component]
pub fn ModelLoading(loaded_bytes: u64, total_bytes: u64) -> Element {
    let total = total_bytes.max(1);
    let percent = (loaded_bytes as f32 / total as f32 * 100.0).min(100.0);
    let downloading = loaded_bytes < total_bytes;

    rsx! {
        div { class: "space-y-3 max-w-md mx-auto", role: "status",
            h3 { class: "text-lg font-semibold text-gray-900 dark:text-gray-100",
                if downloading {
                    "Downloading the model..."
                } else {
                    "Preparing the model..."
                }
            }
            if downloading {
                ProgressBar {
                    value: percent,
                    label: format!("{} of {}", ByteSize(loaded_bytes), ByteSize(total_bytes)),
                }
                p { class: "text-sm text-gray-600 dark:text-gray-400",
                    "The model is saved in this browser, so this only happens once."
                }
            } else {
                p { class: "text-sm text-gray-600 dark:text-gray-400",
                    "Loading the model onto the GPU. Large models can take a minute."
                }
            }
        }
    }
}
