//! The Purr user interface, shared by the desktop and web builds.
use dioxus::prelude::*;

/// Theme and transparency preferences on the document root
pub mod appearance;
/// The design-system components
pub mod components;
/// Saved transcriptions: building entries, searching them and the root store
pub mod history;
/// Custom hooks for state management and UI patterns
pub mod hooks;
/// Routing and session state
mod main_app;
/// Model names, families, states and filtering
pub mod model_catalog;
/// Platform abstraction layer for client-only architecture
pub mod platform;
/// The single owner and writer of the user's settings
pub mod settings_store;
/// Keyboard shortcut registry
pub mod shortcuts;
/// Choosing a file, running a transcription and the result
pub mod transcribe;
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
use hooks::use_models::use_models_provider;
use hooks::use_system_probe::use_system_probe_provider;
use main_app::MainApp;
use settings_store::{status::LoadState, use_settings_store_provider};
use shortcuts::use_shortcut_provider;
use views::settings::dialog_handle::use_settings_dialog_provider;

const FAVICON: Asset = asset!("/assets/icons/purr-mark.svg");
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
    // document::Style compares its children by node identity, so a fresh node per render warns
    let font_face = use_hook(|| {
        let css = inter_font_face();
        rsx! { "{css}" }
    });
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
        document::Style { children: font_face }
        document::Link { rel: "stylesheet", href: DESIGN_CSS }
        document::Link { rel: "stylesheet", href: SCREENS_CSS }
    }
}

/// The root component: providers, the app window and its overlays.
#[component]
pub fn App() -> Element {
    if let Some(error) = platform::platform_init_error() {
        return rsx! {
            Head {}
            div { class: "app",
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
    let settings = use_settings_store_provider(toasts);
    use_appearance_provider(settings);
    use_models_provider();
    use_system_probe_provider();
    hooks::launch_defaults::use_launch_defaults();
    history::store::use_history_provider();
    let session = transcribe::session::use_transcribe_session_provider();
    platform::playback_source::use_audio_route(session);
    hooks::leave_guard::use_leave_guard_during_runs();
    use_layer_host_provider();
    use_settings_dialog_provider();
    use_shortcut_provider();
    let loading = settings.load_state() == LoadState::Loading;

    rsx! {
        Head {}
        div { class: "app",
            if !loading {
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
            }
            LayerOutlet {}
            ToastRegion {}
        }
    }
}
