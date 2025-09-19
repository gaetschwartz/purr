use crate::Route;
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
        nav { class: "bg-white/95 backdrop-blur-sm shadow-sm border-b border-gray-200 sticky top-0 z-50",
            div { class: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8",
                div { class: "flex justify-between items-center h-16",
                    // Logo/Brand
                    div { class: "flex-shrink-0 flex items-center",
                        Link {
                            to: Route::Home {},
                            class: "flex items-center space-x-2 text-xl font-bold text-gray-900 hover:text-teal-600 transition-colors",
                            // Logo icon
                            svg {
                                class: "w-8 h-8 text-teal-600",
                                fill: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    d: "M12 14l9-5-9-5-9 5 9 5z M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z"
                                }
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
