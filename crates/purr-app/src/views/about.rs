use crate::{
    components::{
        brand_mark::BrandMark,
        button::{Button, ButtonVariant},
        sheet::Sheet,
    },
    main_app::Route,
    platform::IS_WEB,
};
use dioxus::prelude::*;
use purr_common::version::Release;

/// The version this build was made from.
pub fn version() -> String {
    Release {
        crate_version: env!("CARGO_PKG_VERSION"),
        channel: purr_common::build_channel!(),
    }
    .to_string()
}

/// Where this build runs.
pub const PLATFORM: &str = if IS_WEB { "Web" } else { "Desktop" };

/// What Purr is, which version this is, where its data lives, and the way to its licences.
#[component]
pub fn AboutSheet(on_close: EventHandler<()>) -> Element {
    let navigator = use_navigator();
    let (place, network) = if IS_WEB {
        (
            "in this browser",
            "Besides loading this page, the only network request",
        )
    } else {
        ("on this computer", "The only network request")
    };
    let version = version();
    rsx! {
        Sheet {
            title: "About Purr",
            on_close,
            actions: rsx! {
                Button {
                    variant: ButtonVariant::Plain,
                    onclick: move |_| {
                        on_close.call(());
                        navigator.push(Route::Licenses {});
                    },
                    "Licenses"
                }
                Button {
                    variant: ButtonVariant::Primary,
                    autofocus: true,
                    onclick: move |_| on_close.call(()),
                    "Done"
                }
            },
            div { class: "about-id",
                BrandMark { large: true }
                div {
                    strong { "Purr" }
                    p { class: "foot", "{version} · {PLATFORM}" }
                }
            }
            p { "Purr turns speech into text with Whisper, running on your own device." }
            p {
                "Audio, transcripts, models, settings and logs stay {place}. {network} is a model download from Hugging Face."
            }
        }
    }
}
