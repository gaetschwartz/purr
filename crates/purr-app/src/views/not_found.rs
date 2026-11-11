use crate::{
    components::{
        button::{Button, ButtonVariant},
        empty_state::EmptyState,
        icon::IconName,
        page::Page,
    },
    main_app::Route,
};
use dioxus::prelude::*;

/// Shown for addresses that match no view.
#[component]
pub fn NotFound(segments: Vec<String>) -> Element {
    let navigator = use_navigator();
    let path = format!("/{}", segments.join("/"));
    rsx! {
        Page { title: "Page not found",
            EmptyState {
                icon: IconName::Info,
                title: "Nothing lives at {path}",
                actions: rsx! {
                    Button {
                        variant: ButtonVariant::Primary,
                        onclick: move |_| {
                            navigator.push(Route::Home {});
                        },
                        "Go to Transcribe"
                    }
                },
                "The link may be out of date."
            }
        }
    }
}
