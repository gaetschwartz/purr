use crate::{
    components::icon::{Icon, IconName, IconSize},
    main_app::Route,
};
use dioxus::prelude::*;
use purr_common::history::entry::HistoryId;

/// A calm way back to the transcription running in the background.
#[component]
pub fn RunLink(id: HistoryId) -> Element {
    rsx! {
        Link {
            class: "run-link surface",
            id: "run-link",
            to: Route::Transcription { id },
            span { class: "spinner", aria_hidden: "true" }
            span { class: "grow", "Transcription in progress" }
            Icon { name: IconName::ChevronRight, size: IconSize::Small }
        }
    }
}
