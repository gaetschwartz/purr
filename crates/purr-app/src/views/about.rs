use crate::{
    components::{
        button::{Button, ButtonVariant},
        sheet::Sheet,
    },
    platform::IS_WEB,
};
use dioxus::prelude::*;
use purr_common::version::Release;

const FONT_LICENCE: &str = include_str!("../../assets/fonts/OFL.txt");

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

/// Where Purr keeps its data, the one network request it makes, and the version.
#[component]
pub fn AboutSheet(on_close: EventHandler<()>) -> Element {
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
            title: "Runs locally",
            on_close,
            actions: rsx! {
                Button {
                    variant: ButtonVariant::Primary,
                    autofocus: true,
                    onclick: move |_| on_close.call(()),
                    "Done"
                }
            },
            p {
                "Audio, transcripts, models, settings and logs stay {place}. {network} is a model download from Hugging Face."
            }
            p { class: "foot", "Purr {version} · {PLATFORM}" }
            details { class: "foot",
                summary { "Inter font licence (SIL OFL)" }
                pre { class: "preview", "{FONT_LICENCE}" }
            }
        }
    }
}
