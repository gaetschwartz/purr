/// Library exports for purr-ui
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
/// Utilities module
mod utils;
/// Define a views module that contains the UI for all Layouts and Routes for our app.
mod views;

use main_app::MainApp;
use theme::{BaseTheme, ColorTokens, CustomTheme, StorageType, Theme, ThemeProvider};


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
    // Log when the app renders
    tracing::debug!("App component rendering with theme system");

    // Setup custom themes as specified
    let solarized_theme = CustomTheme {
        name: "solarized".to_string(),
        base: BaseTheme::Light,
        tokens: ColorTokens {
            primary: "#268bd2".to_string(),
            secondary: "#2aa198".to_string(),
            background: "#fdf6e3".to_string(),
            text: "#657b83".to_string(),
            error: Some("#dc322f".to_string()),
            warning: Some("#cb4b16".to_string()),
            success: Some("#859900".to_string()),
        },
    };

    let custom_themes = vec![solarized_theme];

    // The `rsx!` macro lets us define HTML inside of rust. It expands to an Element with all of our HTML inside.
    rsx! {
        // Global document head elements
        document::Link { rel: "icon", href: FAVICON }
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        document::Link { rel: "stylesheet", href: TAILWIND_CSS }

        // Wrap everything in ThemeProvider as specified
        ThemeProvider {
            default_theme: Theme::System,
            storage_type: StorageType::LocalStorage,
            storage_name: "theme".to_string(),
            custom_themes: Some(custom_themes),
            MainApp {}
        }
    }
}
