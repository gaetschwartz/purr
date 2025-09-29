//! Main application component wrapped with theme provider
//!
//! This module contains the MainApp component that handles the core routing
//! and layout, separate from the theme provider wrapper.

use crate::{
    components::ThemeDemo,
    views::{Home, Logs, Navbar, Settings, Transcription},
};
use dioxus::prelude::*;
use purr_common::platform::FileId;

/// The Route enum defines the structure of internal routes in our app
#[derive(Debug, Clone, Routable, PartialEq)]
#[rustfmt::skip]
pub enum Route {
    #[layout(Navbar)]
        #[route("/")]
        Home {},
        #[route("/transcription/:file")]
        Transcription { file: FileId },
        #[route("/settings")]
        Settings {},
        #[route("/logs")]
        Logs {},
        #[route("/theme-demo")]
        ThemeDemoRoute {},
}

/// Main application component (without theme provider)
///
/// This component contains the core routing and navigation logic.
/// It should be wrapped by the ThemeProvider in the main App component.
#[component]
pub fn MainApp() -> Element {
    // Log when the app renders
    tracing::debug!("MainApp component rendering");

    rsx! {
        // The router component renders the route enum we defined above
        Router::<Route> {}
    }
}

/// Theme demo route component
#[component]
fn ThemeDemoRoute() -> Element {
    rsx! {
        ThemeDemo {}
    }
}