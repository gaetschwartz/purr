use crate::components::{
    banner::{Banner, BannerTone},
    button::{Button, ButtonSize},
};
use dioxus::prelude::*;
use purr_common::history::listing::UnreadableEntry;

const SHOWN: usize = 3;

/// A non-blocking note about stored entries that could not be read.
#[component]
pub fn UnreadableNotice(unreadable: Vec<UnreadableEntry>) -> Element {
    let mut dismissed = use_signal(|| false);
    if unreadable.is_empty() || dismissed() {
        return rsx! {};
    }
    let count = unreadable.len();
    let title = if count == 1 {
        "1 saved transcription can't be read".to_owned()
    } else {
        format!("{count} saved transcriptions can't be read")
    };
    rsx! {
        Banner {
            tone: BannerTone::Warn,
            title,
            role: "status",
            actions: rsx! {
                Button {
                    size: ButtonSize::Small,
                    id: "unreadable-dismiss",
                    onclick: move |_| dismissed.set(true),
                    "Dismiss"
                }
            },
            "The rest of your history is fine. Clear all history removes the ones below."
            ul { style: "margin:6px 0 0;padding-left:18px",
                for entry in unreadable.iter().take(SHOWN) {
                    li { key: "{entry.name}", "{entry.name}: {entry.reason}" }
                }
                if count > SHOWN {
                    li { "and {count - SHOWN} more" }
                }
            }
        }
    }
}
