use super::{
    changes::report, developer_models::DeveloperModelsGroup,
    developer_session_options::SessionOptionsGroup, dialog_handle::use_settings_dialog,
};
use crate::{
    components::{
        group::{Group, Item, ItemAction},
        switch::Switch,
        toast::use_toasts,
    },
    hooks::use_developer_panel::use_developer_panel,
    platform::IS_WEB,
    settings_store::use_settings,
};
use dioxus::prelude::*;

/// Developer mode: its switch and, while it is on, the tools it unlocks.
#[component]
pub fn DeveloperGroup() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let dialog = use_settings_dialog();
    let panel = use_developer_panel();
    let enabled = settings.with(|settings| settings.developer.enabled);

    rsx! {
        Group {
            header: "Developer",
            footer: "Loads models outside the verified catalog and shows live memory and device readings.",
            Item {
                title: "Developer mode",
                control: rsx! {
                    Switch {
                        label: "Developer mode",
                        checked: enabled,
                        onchange: move |on: bool| {
                            report(toasts, settings.update(|settings| settings.set_developer_mode(on)));
                        },
                    }
                },
            }
            if enabled {
                Item {
                    title: "Developer panel",
                    subtitle: "Live readings, refreshed every second.",
                    action: ItemAction::Press(
                        EventHandler::new(move |_| {
                            dialog.close();
                            panel.open();
                        }),
                    ),
                }
            }
        }
        if enabled {
            if IS_WEB {
                SessionOptionsGroup {}
            }
            DeveloperModelsGroup {}
        }
    }
}
