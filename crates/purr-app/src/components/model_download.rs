use super::{
    button::{Button, ButtonSize},
    progress_bar::{Progress, ProgressBar},
};
use crate::model_catalog::{activity::DownloadProgress, format};
use dioxus::prelude::*;

/// A running download: a progress bar with bytes, speed and time left, and a Cancel button.
#[component]
pub fn ModelDownload(
    progress: DownloadProgress,
    #[props(into)] label: String,
    on_cancel: EventHandler<()>,
) -> Element {
    let transferred = format::transferred(&progress);
    let rate = format::rate_and_eta(&progress);
    let bar = progress
        .fraction()
        .map_or(Progress::Indeterminate, Progress::Determinate);
    let value_text = format::percent(&progress).map_or_else(
        || transferred.clone(),
        |percent| format!("{percent}%, {transferred}"),
    );
    rsx! {
        div { class: "dl",
            ProgressBar {
                progress: bar,
                label: "Downloading {label}",
                value_text,
            }
            div { class: "dlmeta",
                span { "{transferred}" }
                span { "{rate}" }
            }
            div { class: "row", style: "justify-content:flex-end",
                Button {
                    size: ButtonSize::Small,
                    onclick: move |_| on_cancel.call(()),
                    "Cancel download"
                }
            }
        }
    }
}
