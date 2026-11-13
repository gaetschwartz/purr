use super::progress_bar::{Progress, ProgressBar};
use crate::model_catalog::{activity::VerifyProgress, format};
use dioxus::prelude::*;

/// A running check of a model's files: a progress bar with `text` under it.
#[component]
pub fn ModelCheck(
    progress: VerifyProgress,
    #[props(into)] label: String,
    #[props(into)] text: String,
) -> Element {
    let bar = progress
        .fraction()
        .map_or(Progress::Indeterminate, Progress::Determinate);
    let value_text = format::check_percent(&progress).map(|percent| format!("{percent}%"));
    rsx! {
        div { class: "dl", role: "status",
            ProgressBar {
                progress: bar,
                label: "Check of {label}",
                value_text,
                thin: true,
            }
            div { class: "dlmeta",
                span { "{text}" }
            }
        }
    }
}
