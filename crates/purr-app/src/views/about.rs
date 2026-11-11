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

const FONT_LICENCE: Asset = asset!("/assets/fonts/OFL.txt");

/// What Purr keeps on this device and the one thing that uses the network.
#[component]
pub fn AboutSheet(on_close: EventHandler<()>) -> Element {
    let (platform, history_place, network) = if IS_WEB {
        (
            "Web",
            "Saved in this browser's storage for this site.",
            "Downloading models, over HTTPS, when you ask for one. The web page itself is loaded from its host.",
        )
    } else {
        (
            "Desktop",
            "Saved in Purr's app data folder on this computer.",
            "Downloading models, over HTTPS, when you ask for one.",
        )
    };
    let version = env!("CARGO_PKG_VERSION");
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
                p { class: "foot", "Purr {version} · {platform}" }
            }
            div { class: "group",
                Item {
                    icon: IconName::Music,
                    tile: TileColor::Green,
                    title: "Audio files",
                    subtitle: "Read from your disk and processed in memory. Never uploaded.",
                }
                Item {
                    icon: IconName::Doc,
                    tile: TileColor::Green,
                    title: "Transcripts and history",
                    subtitle: history_place,
                }
                Item {
                    icon: IconName::Cube,
                    tile: TileColor::Green,
                    title: "Models, settings and logs",
                    subtitle: "Stored locally. Logs never contain transcript text.",
                }
                Item {
                    icon: IconName::Download,
                    tile: TileColor::Blue,
                    title: "The only network use",
                    subtitle: network,
                }
            }
            p { class: "foot",
                "Text is set in Inter, under the "
                if IS_WEB {
                    a { href: FONT_LICENCE, target: "_blank", rel: "noopener", "SIL Open Font License" }
                } else {
                    "SIL Open Font License"
                }
                "."
            }
        }
    }
}
