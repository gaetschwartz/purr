/// Library exports for purr-app
/// This allows both the WASM frontend and native server to use the same components
use dioxus::prelude::*;

/// Define a components module that contains all shared components for our app.
mod components;
/// Custom hooks for state management and UI patterns
pub mod hooks;
/// Main application component
mod main_app;
/// Platform abstraction layer for client-only architecture
pub mod platform;
/// Theme system for managing colors and styling
pub mod theme;
mod transcript;
/// Utilities module
mod utils;
/// Define a views module that contains the UI for all Layouts and Routes for our app.
mod views;

use main_app::MainApp;
use theme::{ColorTokens, CustomTheme, StorageType, Theme, ThemeProvider};

// We can import assets in dioxus with the `asset!` macro. This macro takes a path to an asset relative to the crate root.
// The macro returns an `Asset` type that will display as the path to the asset in the browser or a local path in desktop bundles.
const FAVICON: Asset = asset!("/assets/favicon.ico");
// The asset macro also minifies some assets like CSS and JS to make bundled smaller
const MAIN_CSS: Asset = asset!("/assets/styling/main.css");
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");

/// App is the main component of our app wrapped with theme provider.
/// This component sets up the theme system with custom themes including Solarized.
#[component]
pub fn App() -> Element {
    if let Some(error) = platform::platform_init_error() {
        return rsx! {
            document::Title { "Purr" }
            main { class: "min-h-screen flex items-center justify-center p-6 bg-gray-50 text-gray-900",
                div { class: "max-w-xl space-y-3", role: "alert",
                    h1 { class: "text-2xl font-bold text-red-700", "Purr could not start" }
                    p { "{error}" }
                    p { "Reload the page to try again. If it keeps failing, check that this browser allows site data (IndexedDB and local storage)." }
                }
            }
        };
    }

    // Setup custom themes as specified
    let solarized_theme = CustomTheme {
        name: "solarized".to_owned(),
        base: Some(Theme::Light),
        tokens: ColorTokens {
            primary: "#268bd2".to_owned(),
            secondary: "#2aa198".to_owned(),
            background: "#fdf6e3".to_owned(),
            text: "#657b83".to_owned(),
            error: Some("#dc322f".to_owned()),
            warning: Some("#cb4b16".to_owned()),
            success: Some("#859900".to_owned()),
        },
    };

    let custom_themes = vec![solarized_theme];

    // The `rsx!` macro lets us define HTML inside of rust. It expands to an Element with all of our HTML inside.
    rsx! {
        // Global document head elements
        document::Title { "Purr" }
        document::Link { rel: "icon", href: FAVICON }
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        document::Link { rel: "stylesheet", href: TAILWIND_CSS }

        // Wrap everything in ThemeProvider as specified
        ThemeProvider {
            default_theme: Theme::System,
            storage_type: StorageType::LocalStorage,
            storage_name: "theme".to_owned(),
            custom_themes: Some(custom_themes),
            ErrorBoundary {
                handle_error: |errors: ErrorContext| rsx! {
                    main { class: "min-h-screen flex items-center justify-center p-6 bg-gray-50 text-gray-900 dark:bg-gray-900 dark:text-gray-100",
                        div { class: "max-w-xl space-y-3", role: "alert",
                            h1 { class: "text-2xl font-bold text-red-700 dark:text-red-400", "Something went wrong" }
                            if let Some(error) = errors.error() {
                                p { "{error}" }
                            }
                            button {
                                class: "px-4 py-2 rounded-lg bg-teal-500 text-white hover:bg-teal-600",
                                onclick: move |_| errors.clear_errors(),
                                "Try again"
                            }
                        }
                    }
                },
                MainApp {}
            }
        }
    }
}
