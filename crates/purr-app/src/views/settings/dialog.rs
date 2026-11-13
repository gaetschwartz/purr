use super::{
    advanced::AdvancedPage,
    dialog_handle::use_settings_dialog,
    general::GeneralPage,
    hardware::HardwarePage,
    saved_tick::SavedTick,
    settings_page::{SettingsPage, TabStep},
    transcription::TranscriptionPage,
};
use crate::{
    components::{pages_dialog::PagesDialog, router_bridge::RouterBridge},
    main_app::Route,
    platform::IS_WEB,
};
use dioxus::prelude::*;

const PANEL_ID: &str = "settings-panel";

const SCROLL_TO_TOP: &str =
    "document.querySelector('.pdlg-scroll')?.scrollTo({ top: 0, behavior: 'instant' });";

fn focus_script(page: SettingsPage) -> String {
    format!("document.getElementById({:?})?.focus();", page.tab_id())
}

fn step_for(key: &Key) -> Option<TabStep> {
    match key {
        Key::Named(NamedKey::ArrowDown | NamedKey::ArrowRight) => Some(TabStep::Next),
        Key::Named(NamedKey::ArrowUp | NamedKey::ArrowLeft) => Some(TabStep::Previous),
        Key::Named(NamedKey::Home) => Some(TabStep::First),
        Key::Named(NamedKey::End) => Some(TabStep::Last),
        Key::Named(_) | Key::Character(_) => None,
    }
}

/// The Settings dialog: the page list and the selected page. Render it while the dialog is open.
///
/// It closes itself when the route changes under it, as it does when a picker inside sends the
/// user to another screen.
#[component]
pub fn SettingsDialogView() -> Element {
    let dialog = use_settings_dialog();
    let route = use_route::<Route>();
    let opened_on = use_hook(|| route.clone());
    let left = route != opened_on;
    use_effect(use_reactive!(|left| {
        if left {
            dialog.close();
        }
    }));

    let pages = SettingsPage::visible(IS_WEB);
    let page = dialog.page();
    let select = move |next: SettingsPage| {
        dialog.select(next);
        document::eval(SCROLL_TO_TOP);
    };
    let navigate = {
        let pages = pages.clone();
        move |event: KeyboardEvent| {
            let Some(step) = step_for(&event.key()) else {
                return;
            };
            event.prevent_default();
            let next = dialog.page().step(&pages, step);
            dialog.select(next);
            document::eval(&focus_script(next));
            document::eval(SCROLL_TO_TOP);
        }
    };

    rsx! {
        PagesDialog {
            title: "Settings",
            on_close: move |()| dialog.close(),
            actions: rsx! {
                SavedTick {}
            },
            nav: rsx! {
                div {
                    class: "pdlg-nav",
                    role: "tablist",
                    aria_orientation: "vertical",
                    aria_label: "Settings pages",
                    onkeydown: navigate,
                    for entry in pages {
                        button {
                            key: "{entry.tab_id()}",
                            id: entry.tab_id(),
                            class: "pdlg-tab",
                            r#type: "button",
                            role: "tab",
                            aria_selected: entry == page,
                            aria_controls: PANEL_ID,
                            tabindex: if entry == page { "0" } else { "-1" },
                            autofocus: entry == page,
                            onclick: move |_| select(entry),
                            "{entry.label()}"
                        }
                    }
                }
            },
            RouterBridge {
                div {
                    id: PANEL_ID,
                    class: "pdlg-panel",
                    role: "tabpanel",
                    aria_labelledby: page.tab_id(),
                    key: "{page.tab_id()}",
                    match page {
                        SettingsPage::General => rsx! {
                            GeneralPage {}
                        },
                        SettingsPage::Transcription => rsx! {
                            TranscriptionPage {}
                        },
                        SettingsPage::Hardware => rsx! {
                            HardwarePage {}
                        },
                        SettingsPage::Advanced => rsx! {
                            AdvancedPage {}
                        },
                    }
                }
            }
        }
    }
}
