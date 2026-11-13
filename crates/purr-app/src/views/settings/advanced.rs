use super::{changes::report, dialog_handle::use_settings_dialog, saved_audio::SavedAudioGroup};
use crate::{
    components::{
        dialog::{Dialog, DialogTone},
        group::{Group, Item, ItemAction},
        presence::Presence,
        select::{Select, SelectOption},
        toast::{Toast, ToastKind, use_toasts},
    },
    main_app::Route,
    platform::IS_WEB,
    settings_store::use_settings,
    views::{
        about::{AboutSheet, PLATFORM, version},
        shortcuts_sheet::ShortcutsSheet,
    },
};
use dioxus::prelude::*;
use purr_common::settings::LogLevel;

/// Logs, shortcuts, the version and the reset.
#[component]
pub fn AdvancedPage() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let dialog = use_settings_dialog();
    let navigator = use_navigator();
    let mut keys_open = use_signal(|| false);
    let mut about_open = use_signal(|| false);
    let mut reset_open = use_signal(|| false);
    let level = settings.with(|settings| settings.logging.level.clone());
    let version = version();

    let reset_settings = move |()| {
        reset_open.set(false);
        let previous = settings.reset_to_defaults();
        toasts.show(
            Toast::new(ToastKind::Info, "Settings reset").with_action("Undo", move || {
                if let Err(error) = settings.replace(previous.clone()) {
                    toasts.error("Couldn't undo the reset", error.to_string());
                }
            }),
        );
    };

    rsx! {
        Group {
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
                action: ItemAction::Press(
                    EventHandler::new(move |_| {
                        dialog.close();
                        navigator.push(Route::Logs {});
                    }),
                ),
            }
        }
        Group {
            Item {
                title: "Keyboard shortcuts",
                value: "?",
                action: ItemAction::Press(EventHandler::new(move |_| keys_open.set(true))),
            }
            Item {
                title: "Purr {version} · {PLATFORM}",
                action: ItemAction::Press(EventHandler::new(move |_| about_open.set(true))),
            }
        }
        if IS_WEB {
            SavedAudioGroup {}
        }
        Group {
            Item {
                title: "Reset all settings",
                subtitle: "Models and history are kept.",
                danger: true,
                action: ItemAction::Press(EventHandler::new(move |_| reset_open.set(true))),
            }
        }
        Presence { present: keys_open(),
            ShortcutsSheet { on_close: move |()| keys_open.set(false) }
        }
        Presence { present: about_open(),
            AboutSheet { on_close: move |()| about_open.set(false) }
        }
        Presence { present: reset_open(),
            Dialog {
                title: "Reset all settings?",
                confirm_label: "Reset settings",
                tone: DialogTone::Destructive,
                on_confirm: reset_settings,
                on_cancel: move |()| reset_open.set(false),
                "Models and history are kept."
            }
        }
    }
}
