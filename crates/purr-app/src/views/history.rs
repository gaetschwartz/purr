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

/// Past transcriptions; a placeholder until history is stored.
#[component]
pub fn History() -> Element {
    let navigator = use_navigator();
    rsx! {
        Page { title: "History", wide: true,
            EmptyState {
                icon: IconName::Cat,
                title: "No transcriptions yet",
                actions: rsx! {
                    Button {
                        variant: ButtonVariant::Primary,
                        icon: IconName::Plus,
                        onclick: move |_| {
                            navigator.push(Route::Home {});
                        },
                        "Transcribe a file"
                    }
                },
                "Transcripts you make are saved here, on this device only. Nothing is synced or uploaded."
            }
        }
    }
}
