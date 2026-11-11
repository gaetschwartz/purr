use super::progress_bar::{Progress, ProgressBar};
use crate::{model_catalog::listing::Summary, utils::byte_size::ByteSize};
use dioxus::prelude::*;

/// What the third stat card says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Headline {
    /// A download is running: the model's label and how far it is, if known.
    Downloading { label: String, percent: Option<u32> },
    /// The model transcriptions use, by label.
    InUse(String),
}

/// The three stat cards above the model list: stored models, space used, and what is going on.
#[component]
pub fn ModelStats(summary: Summary, headline: Headline) -> Element {
    let (title, text) = match headline {
        Headline::Downloading { label, percent } => (
            "Downloading",
            percent.map_or_else(|| label.clone(), |percent| format!("{label} · {percent}%")),
        ),
        Headline::InUse(label) => ("Used for transcription", label),
    };
    rsx! {
        div { class: "mstats",
            div { class: "mstat glass",
                div { class: "k", "Downloaded" }
                div { class: "v",
                    "{summary.installed} "
                    span { class: "faint", style: "font-size:14px;font-weight:500",
                        "of {summary.total}"
                    }
                }
            }
            div { class: "mstat glass",
                div { class: "k", "Space used" }
                div { class: "v", "{ByteSize(summary.used_bytes)}" }
                ProgressBar {
                    progress: Progress::Determinate(summary.used_fraction()),
                    label: "Share of all models stored on this device",
                    value_text: "{ByteSize(summary.used_bytes)} of {ByteSize(summary.catalog_bytes)}",
                    thin: true,
                }
            }
            div { class: "mstat glass",
                div { class: "k", "{title}" }
                div {
                    class: "v",
                    style: "font-size:17px;line-height:30px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis",
                    title: "{text}",
                    "{text}"
                }
            }
        }
    }
}
