use crate::{
    components::{Icon, IconType},
    Route,
};
use dioxus::prelude::*;

const NAVBAR_CSS: Asset = asset!("/assets/styling/navbar.css");

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
                            Icon {
                                icon_type: IconType::Logo,
                            }
                            span { "Purr" }
                        }
                    }

                    // Navigation links
                    div { class: "hidden md:block",
                        div { class: "ml-10 flex items-baseline space-x-4",
                            Link {
                                to: Route::Home {},
                                class: "px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:text-teal-600 hover:bg-teal-50 transition-all duration-200",
                                "Upload Audio"
                            }
                        }
                    }
                }
            }
        }

        // The `Outlet` component is used to render the next component inside the layout
        Outlet::<Route> {}
    }
}
