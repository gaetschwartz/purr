use super::{delete_dialog::DeleteDialog, export_host::ExportHost, rename_dialog::RenameDialog};
use crate::{
    components::{
        icon::{Icon, IconName, IconSize},
        icon_button::IconButton,
        layer::PopoverAlign,
        popover::{Popover, PopoverOption},
    },
    history::wording::{preview_text, row_meta},
    main_app::Route,
    utils::{file_kind::kind_label, local_time::when_text},
};
use dioxus::prelude::*;
use purr_common::history::entry::HistorySummary;

/// One saved transcription: open it, or use its menu to rename, export or delete it.
#[component]
pub fn HistoryRow(summary: HistorySummary, now_ms: i64) -> Element {
    let navigator = use_navigator();
    let mut menu_open = use_signal(|| false);
    let mut renaming = use_signal(|| false);
    let mut exporting = use_signal(|| false);
    let mut deleting = use_signal(|| false);

    let menu_id = format!("hm-{}", summary.id);
    let meta = row_meta(&when_text(summary.created_at_ms, now_ms), &summary);
    let open_id = summary.id.clone();
    let menu_label = format!("More actions for {}", summary.title);

    rsx! {
        div { class: "hrow",
            button {
                class: "open",
                r#type: "button",
                id: "ho-{summary.id}",
                onclick: move |_| {
                    navigator.push(Route::HistoryItem { id: open_id.clone() });
                },
                span { class: "badge", aria_hidden: "true", {kind_label(&summary.file_name)} }
                span { class: "grow",
                    span { class: "nm", "{summary.title}" }
                    span { class: "pv", "{preview_text(&summary)}" }
                    span { class: "mt", "{meta}" }
                }
            }
            IconButton {
                icon: IconName::More,
                label: menu_label.clone(),
                id: menu_id.clone(),
                aria_haspopup: "menu",
                aria_expanded: menu_open(),
                onclick: move |_| menu_open.toggle(),
            }
        }
        if menu_open() {
            Popover {
                anchor: menu_id,
                label: menu_label,
                menu: true,
                align: PopoverAlign::End,
                on_close: move |()| menu_open.set(false),
                div { class: "plist", role: "menu", aria_label: "Actions for {summary.title}",
                    PopoverOption {
                        title: "Open",
                        leading: rsx! {
                            Icon { name: IconName::Doc, size: IconSize::Small }
                        },
                        onselect: {
                            let id = summary.id.clone();
                            move |()| {
                                menu_open.set(false);
                                navigator.push(Route::HistoryItem { id: id.clone() });
                            }
                        },
                    }
                    PopoverOption {
                        title: "Rename…",
                        leading: rsx! {
                            Icon { name: IconName::Pencil, size: IconSize::Small }
                        },
                        onselect: move |()| {
                            menu_open.set(false);
                            renaming.set(true);
                        },
                    }
                    PopoverOption {
                        title: "Export…",
                        leading: rsx! {
                            Icon { name: IconName::Share, size: IconSize::Small }
                        },
                        onselect: move |()| {
                            menu_open.set(false);
                            exporting.set(true);
                        },
                    }
                    PopoverOption {
                        title: "Delete…",
                        danger: true,
                        leading: rsx! {
                            Icon { name: IconName::Trash, size: IconSize::Small }
                        },
                        onselect: move |()| {
                            menu_open.set(false);
                            deleting.set(true);
                        },
                    }
                }
            }
        }
        if renaming() {
            RenameDialog {
                id: summary.id.clone(),
                title: summary.title.clone(),
                on_close: move |()| renaming.set(false),
            }
        }
        if exporting() {
            ExportHost { id: summary.id.clone(), on_close: move |()| exporting.set(false) }
        }
        if deleting() {
            DeleteDialog {
                id: summary.id.clone(),
                title: summary.title.clone(),
                on_close: move |()| deleting.set(false),
            }
        }
    }
}
