use super::sections::SettingsSection;
use crate::{
    components::{
        group::{Group, Item, ItemAction, TileColor},
        icon::IconName,
    },
    views::{
        about::{AboutSheet, PLATFORM, VERSION},
        shortcuts_sheet::ShortcutsSheet,
    },
};
use dioxus::prelude::*;

/// Privacy, the version, and the reset button.
#[component]
pub fn AboutSection(on_reset: EventHandler<()>) -> Element {
    let mut privacy_open = use_signal(|| false);
    let mut keys_open = use_signal(|| false);

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
                    icon: IconName::Keyboard,
                    tile: TileColor::Gray,
                    title: "Keyboard shortcuts",
                    value: "?",
                    action: ItemAction::Press(EventHandler::new(move |_| keys_open.set(true))),
                }
                Item {
                    icon: IconName::Cat,
                    tile: TileColor::Indigo,
                    title: "Purr {VERSION}",
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
        if keys_open() {
            ShortcutsSheet { on_close: move |()| keys_open.set(false) }
        }
    }
}
