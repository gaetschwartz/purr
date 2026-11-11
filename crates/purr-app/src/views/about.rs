use crate::{
    components::{
        brand_mark::BrandMark,
        button::{Button, ButtonVariant},
        group::{Item, TileColor},
        icon::IconName,
        sheet::Sheet,
    },
    platform::IS_WEB,
};
use dioxus::prelude::*;

const FONT_LICENCE: &str = include_str!("../../assets/fonts/OFL.txt");

/// The version this build was made from.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Where this build runs.
pub const PLATFORM: &str = if IS_WEB { "Web" } else { "Desktop" };

/// What Purr keeps on this device and the one thing that uses the network.
#[component]
pub fn AboutSheet(on_close: EventHandler<()>) -> Element {
    let (audio, history, stored, network) = if IS_WEB {
        (
            "Read into this browser's memory and decoded by the browser. Never uploaded.",
            "The text and a waveform are saved in this browser's storage for this site. The audio is not saved.",
            "Models and settings are kept in this browser's storage. Logs stay in memory until you close the page and list file names, models and errors, never transcript text.",
            "Downloading a model from Hugging Face over HTTPS when you ask for one. Hugging Face sees which model you fetch, as with any download. The page itself is loaded from its host. There are no update checks, accounts or usage statistics.",
        )
    } else {
        (
            "Read from your disk and decoded on this computer. A temporary copy sits in the system's temporary folder while you work and is deleted when you move on; Purr clears leftovers older than a day when it starts. Never uploaded.",
            "The text and a waveform are saved in Purr's app data folder on this computer. The audio is not copied.",
            "Models and settings are kept in Purr's folders on this computer. Logs stay in memory until you close Purr and list file names, models and errors, never transcript text.",
            "Downloading a model from Hugging Face over HTTPS when you ask for one. Hugging Face sees which model you fetch, as with any download. There are no update checks, accounts or usage statistics.",
        )
    };
    rsx! {
        Sheet {
            title: "Your audio stays here",
            on_close,
            actions: rsx! {
                Button {
                    variant: ButtonVariant::Primary,
                    autofocus: true,
                    onclick: move |_| on_close.call(()),
                    "Done"
                }
            },
            div { class: "row",
                BrandMark { large: true }
                p { class: "foot", "Purr {VERSION} · {PLATFORM}" }
            }
            div { class: "group",
                Item {
                    icon: IconName::Music,
                    tile: TileColor::Green,
                    title: "Audio files",
                    subtitle: audio,
                }
                Item {
                    icon: IconName::Doc,
                    tile: TileColor::Green,
                    title: "Transcripts and history",
                    subtitle: history,
                }
                Item {
                    icon: IconName::Cube,
                    tile: TileColor::Green,
                    title: "Models, settings and logs",
                    subtitle: stored,
                }
                Item {
                    icon: IconName::Download,
                    tile: TileColor::Blue,
                    title: "The only network use",
                    subtitle: network,
                }
            }
            details { class: "foot",
                summary { "Text is set in Inter, under the SIL Open Font License" }
                pre { class: "preview", "{FONT_LICENCE}" }
            }
        }
    }
}
