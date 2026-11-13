use super::sections::SettingsSection;
use crate::{
    components::group::{Group, Item, ItemAction},
    views::{
        about::{AboutSheet, PLATFORM, version},
        shortcuts_sheet::ShortcutsSheet,
    },
};
use dioxus::prelude::*;

/// The keyboard shortcuts, the version, and the reset button.
#[component]
pub fn AboutSection(on_reset: EventHandler<()>) -> Element {
    let mut keys_open = use_signal(|| false);
    let mut about_open = use_signal(|| false);
    let version = version();

    rsx! {
        div { class: "sect", id: SettingsSection::About.anchor(),
            Group { header: SettingsSection::About.label(),
                Item {
                    title: "Keyboard shortcuts",
                    value: "?",
                    action: ItemAction::Press(EventHandler::new(move |_| keys_open.set(true))),
                }
                Item {
                    title: "Purr {version}",
                    value: PLATFORM,
                    action: ItemAction::Press(EventHandler::new(move |_| about_open.set(true))),
                }
            }
            Group { label: "Reset",
                Item {
                    title: "Reset all settings",
                    subtitle: "Models and history are kept.",
                    danger: true,
                    action: ItemAction::Press(EventHandler::new(move |_| on_reset.call(()))),
                }
            }
        }
        if keys_open() {
            ShortcutsSheet { on_close: move |()| keys_open.set(false) }
        }
        if about_open() {
            AboutSheet { on_close: move |()| about_open.set(false) }
        }
    }
}
