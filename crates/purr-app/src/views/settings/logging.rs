use super::{changes::report, sections::SettingsSection};
use crate::{
    components::{
        group::{Group, Item, ItemAction},
        select::{Select, SelectOption},
        toast::use_toasts,
    },
    main_app::Route,
    settings_store::use_settings,
};
use dioxus::prelude::*;
use purr_common::settings::LogLevel;

/// Log level and the way to the Logs screen.
#[component]
pub fn LoggingSection() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let level = settings.with(|settings| settings.logging.level.clone());

    rsx! {
        Group { id: SettingsSection::Logging.anchor(), header: SettingsSection::Logging.label(),
            Item {
                title: "Log level",
                control: rsx! {
                    Select {
                        label: "Log level",
                        value: level,
                        options: vec![
                            SelectOption::new(LogLevel::Error, "Error"),
                            SelectOption::new(LogLevel::Warn, "Warning"),
                            SelectOption::new(LogLevel::Info, "Info"),
                            SelectOption::new(LogLevel::Debug, "Debug"),
                            SelectOption::new(LogLevel::Trace, "Trace"),
                        ],
                        onchange: move |level: LogLevel| report(toasts, settings.set_log_level(level)),
                    }
                },
            }
            Item {
                title: "Open logs",
                action: ItemAction::Navigate(Route::Logs {}),
            }
        }
    }
}
