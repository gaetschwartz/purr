use super::sections::SettingsSection;
use crate::{
    components::{
        group::{Group, Item, ItemAction, TileColor},
        icon::IconName,
    },
    platform::IS_WEB,
    views::about::AboutSheet,
};
use dioxus::prelude::*;

const PLATFORM: &str = if IS_WEB { "Web" } else { "Desktop" };

/// Privacy, the version, and the reset button.
#[component]
pub fn AboutSection(on_reset: EventHandler<()>) -> Element {
    let mut privacy_open = use_signal(|| false);

    rsx! {
        div { class: "sect", id: SettingsSection::About.anchor(),
            Group { header: SettingsSection::About.label(),
                Item {
                    icon: IconName::Shield,
                    tile: TileColor::Green,
                    title: "Privacy",
                    subtitle: "What stays on this device, and the one thing that uses the network.",
                    action: ItemAction::Press(EventHandler::new(move |_| privacy_open.set(true))),
                }
                Item {
                    icon: IconName::Cat,
                    tile: TileColor::Indigo,
                    title: "Purr {env!(\"CARGO_PKG_VERSION\")}",
                    subtitle: PLATFORM,
                }
            }
            Group { label: "Reset",
                Item {
                    title: "Reset all settings",
                    subtitle: "Models and history are not affected.",
                    danger: true,
                    action: ItemAction::Press(EventHandler::new(move |_| on_reset.call(()))),
                }
            }
        }
        if privacy_open() {
            AboutSheet { on_close: move |()| privacy_open.set(false) }
        }
    }
}
