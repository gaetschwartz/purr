use crate::{
    components::{button::Button, empty_state::EmptyState, icon::IconName, page::Page},
    main_app::Route,
};
use dioxus::prelude::*;

/// The model manager; a placeholder until it moves out of Settings.
#[component]
pub fn Models() -> Element {
    let navigator = use_navigator();
    rsx! {
        Page {
            title: "Models",
            subtitle: "Speech models stored on this device. Larger models are more accurate and slower.",
            wide: true,
            EmptyState {
                icon: IconName::Cube,
                title: "Models are managed in Settings for now",
                actions: rsx! {
                    Button {
                        icon: IconName::Sliders,
                        onclick: move |_| {
                            navigator.push(Route::Settings {});
                        },
                        "Open Settings"
                    }
                },
                "Choose, download and remove speech models from the Model section of Settings."
            }
        }
    }
}
