use super::sections::SettingsSection;
use crate::{
    components::group::{Group, Item, ItemAction},
    views::{
        about::{PLATFORM, VERSION},
        shortcuts_sheet::ShortcutsSheet,
    },
};
use dioxus::prelude::*;

/// The keyboard shortcuts, the version, and the reset button.
#[component]
pub fn AboutSection(on_reset: EventHandler<()>) -> Element {
    let mut keys_open = use_signal(|| false);

    rsx! {
        div { class: "sect", id: SettingsSection::About.anchor(),
            Group { header: SettingsSection::About.label(),
                Item {
                    title: "Keyboard shortcuts",
                    value: "?",
                    action: ItemAction::Press(EventHandler::new(move |_| keys_open.set(true))),
                }
                Item { title: "Purr {VERSION}", value: PLATFORM }
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
    }
}
