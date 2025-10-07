use crate::{
    components::{Icon, IconType},
    main_app::Route,
    views::{DragDropZone, NavBarState},
};
use dioxus::prelude::*;

/// The Home page component that will be rendered when the current route is `[Route::Home]`
/// Contains a drop zone to upload a file, if clicked, will open a file dialog
/// If a file is uploaded, it will be sent to the backend to be processed
/// the drop zone should be surrounded by a dashed border
#[component]
pub fn Home() -> Element {
    let mut navbar_state = use_context::<NavBarState>();
    use_effect(move || {
        navbar_state.title.set("Purr".to_string());
        navbar_state.show_back.set(false);
        navbar_state.top_right.set(Some(rsx! {
            Link {
                to: Route::Settings {},
                class: "p-2 rounded-lg text-gray-600 hover:text-teal-600 hover:bg-gray-100 transition-colors",
                title: "Settings",
                onclick: move |_| {
                    tracing::info!("Navigating to Settings page");
                },
                Icon { icon_type: IconType::Settings }
            }
        }));
    });

    // Log when the home page renders
    use_effect(move || {
        tracing::info!("Home page loaded");
        tracing::debug!("Rendering DragDropZone component");
    });

    rsx! {
        DragDropZone {}
    }
}
