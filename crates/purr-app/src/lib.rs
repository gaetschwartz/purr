//! The Purr user interface, shared by the desktop and web builds.
use dioxus::prelude::*;

/// Theme and transparency preferences on the document root
pub mod appearance;
/// The design-system components
pub mod components;
/// Custom hooks for state management and UI patterns
pub mod hooks;
/// Routing and session state
mod main_app;
/// Model names, families, states and filtering
pub mod model_catalog;
/// Platform abstraction layer for client-only architecture
pub mod platform;
/// Keyboard shortcut registry
pub mod shortcuts;
mod transcript;
/// Utilities module
mod utils;
/// The shell layout and the routed views
mod views;

use appearance::use_appearance_provider;
use components::{
    banner::{Banner, BannerTone},
    button::{Button, ButtonVariant},
    icon::IconName,
    layer::{LayerOutlet, use_layer_host_provider},
    page::document_title,
    toast::{ToastRegion, use_toast_provider},
};
use main_app::MainApp;
use shortcuts::use_shortcut_provider;

const FAVICON: Asset = asset!("/assets/icons/purr-mark.svg");
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");
// unminified: the minifier strips the -webkit- prefixed glass fallbacks older Safari needs
const DESIGN_CSS: Asset = asset!(
    "/assets/styling/design.css",
    AssetOptions::css().with_minify(false)
);
const SCREENS_CSS: Asset = asset!("/assets/styling/screens.css");
const INTER: Asset = asset!("/assets/fonts/InterVariable.woff2");

/// The `@font-face` for the bundled Inter, whose URL only exists once the asset is resolved.
fn inter_font_face() -> String {
    format!(
        "@font-face{{font-family:Inter;font-style:normal;font-weight:100 900;font-display:swap;src:url(\"{INTER}\") format(\"woff2\")}}"
    )
}

#[component]
fn Head() -> Element {
    rsx! {
        document::Title { {document_title("Transcribe")} }
        document::Link { rel: "icon", r#type: "image/svg+xml", href: FAVICON }
        document::Link {
            rel: "preload",
            href: INTER,
            r#as: "font",
            r#type: "font/woff2",
            crossorigin: "anonymous",
        }
        document::Style { {inter_font_face()} }
        document::Link { rel: "stylesheet", href: TAILWIND_CSS }
        document::Link { rel: "stylesheet", href: DESIGN_CSS }
        document::Link { rel: "stylesheet", href: SCREENS_CSS }
    }
}

#[component]
fn Ambient() -> Element {
    rsx! {
        div { class: "ambient", aria_hidden: "true",
            i { class: "blob b1" }
            i { class: "blob b2" }
            i { class: "blob b3" }
            i { class: "blob b4" }
        }
    }
}

/// The root component: providers, the app window and its overlays.
#[component]
pub fn App() -> Element {
    if let Some(error) = platform::platform_init_error() {
        return rsx! {
            Head {}
            div { class: "app",
                Ambient {}
                div { class: "scroller",
                    div { class: "content",
                        h1 { class: "lt", "Purr could not start" }
                        Banner { tone: BannerTone::Bad, title: "{error}", role: "alert",
                            "Reload the page to try again. If it keeps failing, check that this browser allows site data (IndexedDB and local storage)."
                        }
                    }
                }
            }
        };
    }

    let toasts = use_toast_provider();
    use_appearance_provider(toasts);
    use_layer_host_provider();
    use_shortcut_provider();

    rsx! {
        Head {}
        div { class: "app",
            Ambient {}
            ErrorBoundary {
                handle_error: |errors: ErrorContext| {
                    let message = errors.error().map(|error| error.to_string());
                    rsx! {
                    div { class: "scroller",
                        div { class: "content",
                            h1 { class: "lt", "Something went wrong" }
                            Banner {
                                tone: BannerTone::Bad,
                                title: "A part of Purr stopped working",
                                role: "alert",
                                actions: rsx! {
                                    Button {
                                        variant: ButtonVariant::Primary,
                                        icon: IconName::Retry,
                                        onclick: move |_| errors.clear_errors(),
                                        "Try again"
                                    }
                                },
                                if let Some(message) = message {
                                    "{message}"
                                }
                            }
                        }
                    }
                    }
                },
                MainApp {}
            }
            LayerOutlet {}
            ToastRegion {}
        }
    }
}
