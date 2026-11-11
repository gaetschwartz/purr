use super::{changes::report, sections::SettingsSection};
use crate::{
    components::{
        group::{Group, Item, ItemAction, TileColor},
        icon::IconName,
        select::{Select, SelectOption},
        switch::Switch,
        toast::use_toasts,
    },
    main_app::Route,
    platform::IS_WEB,
    settings_store::use_settings,
};
use dioxus::prelude::*;
use purr_common::settings::LogLevel;

/// Log level, the log file and the way to the Logs screen.
#[component]
pub fn LoggingSection() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let (level, file_logging) = settings.with(|settings| {
        (
            settings.logging.level.clone(),
            settings.logging.enable_file_logging,
        )
    });

    rsx! {
        Group { id: SettingsSection::Logging.anchor(), header: SettingsSection::Logging.label(),
            Item {
                icon: IconName::Logs,
                tile: TileColor::Gray,
                title: "Log level",
                subtitle: "Debug and Trace help when reporting a problem.",
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
            if !IS_WEB {
                Item {
                    icon: IconName::File,
                    tile: TileColor::Blue,
                    title: "Write logs to a file",
                    subtitle: "This version of Purr does not write a log file yet. Logs are kept in memory, and you can export them from Logs.",
                    control: rsx! {
                        Switch {
                            label: "Write logs to a file",
                            checked: file_logging,
                            onchange: move |value| report(toasts, settings.update(|s| s.logging.enable_file_logging = value)),
                        }
                    },
                }
            }
            Item {
                icon: IconName::Logs,
                tile: TileColor::Indigo,
                title: "Open logs",
                action: ItemAction::Navigate(Route::Logs {}),
            }
        }
    }
}
