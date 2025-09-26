use crate::{
    components::{Icon, IconType},
    Route,
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
    rsx! {
        // Navigation header
        nav { class: "navbar",
            div { class: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8",
                div { class: "flex justify-between items-center h-16",
                    // Logo/Brand
                    div { class: "flex-shrink-0 flex items-center",
                        Link {
                            to: Route::Home {},
                            class: "flex items-center space-x-2 text-xl font-bold text-gray-900 hover:text-teal-600 transition-colors",
                            Icon { icon_type: IconType::Logo }
                            span { "Purr" }
                        }
                    }

                    // Navigation Icons
                    div { class: "flex items-center space-x-4",
                        Link {
                            to: Route::Settings {},
                            class: "p-2 rounded-lg text-gray-600 hover:text-teal-600 hover:bg-gray-100 transition-colors",
                            title: "Settings",
                            onclick: move |_| {
                                tracing::info!("Navigating to Settings page");
                            },
                            Icon { icon_type: IconType::Settings }
                        }
                        Link {
                            to: Route::Logs {},
                            class: "p-2 rounded-lg text-gray-600 hover:text-teal-600 hover:bg-gray-100 transition-colors",
                            title: "Logs",
                            onclick: move |_| {
                                tracing::info!("Navigating to Logs page");
                            },
                            Icon { icon_type: IconType::Logs }
                        }
                    }
                }
            }
        }

        // The `Outlet` component is used to render the next component inside the layout
        Outlet::<Route> {}
    }
}
