use crate::{
    components::{Icon, IconType},
    main_app::Route,
};
use dioxus::prelude::*;

// const NAVBAR_CSS: Asset = asset!("/assets/styling/navbar.css");

/// The Navbar component that will be rendered on all pages of our app since every page is under the layout.
///
///
/// This layout component wraps the UI with a common navbar. The contents of the routes
/// will be rendered under the outlet inside this component
#[component]
pub fn Navbar() -> Element {
    let state = use_context::<NavBarState>();
    let title = state.title;
    let route: Route = use_route();

    rsx! {
        // Navigation header
        nav { class: "navbar",
            div { class: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8",
                div { class: "flex justify-between items-center h-16",
                    if *state.show_back.read() {
                        // Back button
                        Link {
                            to: route.parent().unwrap_or(Route::Home {}),
                            class: "p-2 rounded-lg text-gray-600 hover:text-teal-600 hover:bg-gray-100 transition-colors",
                            title: "Back",
                            onclick: move |_| {
                                tracing::info!("Navigating back to Home page");
                            },
                            Icon { icon_type: IconType::BackArrow }
                        }
                    } else {
                        div { class: "w-10" } // Placeholder to keep title centered
                    }
                    // Logo
                    div { class: "flex-shrink-0 flex items-center",
                        Link {
                            to: Route::Home {},
                            class: "flex items-center space-x-2 text-xl font-bold text-gray-900 hover:text-teal-600 transition-colors",
                            Icon { icon_type: IconType::Logo }
                            span { "{title}" }
                        }
                    }

                    // Navigation Icons
                    div { class: "flex items-center space-x-4", {state.top_right} }
                }
            }
        }

        // The `Outlet` component is used to render the next component inside the layout
        Outlet::<Route> {}
    }
}

#[derive(Clone, Copy)]
pub struct NavBarState {
    pub title: Signal<String>,
    pub show_back: Signal<bool>,
    pub top_right: Signal<Option<Element>>,
}
