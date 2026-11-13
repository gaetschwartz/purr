use super::{count, filter::Entries};
use crate::{
    components::{
        button::Button,
        icon::{Icon, IconName, IconSize},
        layer::PopoverAlign,
        popover::{Popover, PopoverOption},
        presence::Presence,
        toast::{Toasts, use_toasts},
    },
    platform::{
        export::{SaveOutcome, TextFile, save_text_file},
        log_export::{LogExportFormat, format_entries},
        logging::LogEntry,
    },
};
use dioxus::{core::spawn_forever, prelude::*};
use std::rc::Rc;

/// Saves `entries` in `format` and tells the user how it went.
fn start_export(format: LogExportFormat, entries: Entries, toasts: Toasts) {
    spawn_forever(async move {
        let file_name = format.file_name();
        let saved = match format_entries(format, &entries, LogEntry::formatted_timestamp) {
            Ok(content) => {
                save_text_file(TextFile {
                    file_name: &file_name,
                    content_type: format.content_type(),
                    content: &content,
                })
                .await
            }
            Err(error) => Err(error),
        };
        match saved {
            Ok(SaveOutcome::Saved) => {
                toasts.success(format!("Saved {file_name}"), count::entries(entries.len()));
            }
            Ok(SaveOutcome::Cancelled) => {}
            Err(error) => {
                tracing::error!("Failed to export the logs: {error}");
                toasts.error("Couldn't export the logs", error.to_string());
            }
        }
    });
}

/// The Export button and its menu of file formats; it saves every entry in `entries`.
#[component]
pub fn ExportMenu(entries: Entries) -> Element {
    let mut open = use_signal(|| false);
    let toasts = use_toasts();

    rsx! {
        Button {
            id: "log-export",
            icon: IconName::Share,
            disabled: entries.is_empty(),
            aria_haspopup: "menu",
            aria_expanded: open(),
            onclick: move |_| open.toggle(),
            "Export"
        }
        Presence { present: open(),
            Popover {
                anchor: "log-export",
                label: "Export logs",
                menu: true,
                align: PopoverAlign::End,
                on_close: move |()| open.set(false),
                div { class: "plist", role: "menu", aria_label: "Export logs",
                    for format in LogExportFormat::ALL {
                        PopoverOption {
                            key: "{format.label()}",
                            title: format.label(),
                            trailing: format.extension(),
                            leading: rsx! {
                                Icon { name: IconName::Doc, size: IconSize::Small }
                            },
                            onselect: {
                                let entries = Rc::clone(&entries);
                                move |()| {
                                    open.set(false);
                                    start_export(format, Rc::clone(&entries), toasts);
                                }
                            },
                        }
                    }
                }
            }
        }
    }
}
