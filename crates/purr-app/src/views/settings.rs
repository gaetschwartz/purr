//! The Settings view: every setting in inset grouped lists, saved as it changes.

mod about;
mod appearance;
mod browser_panel;
mod changes;
mod hugging_face_token;
pub mod jump;
mod logging;
mod performance;
mod saved_tick;
mod section_nav;
pub mod sections;
mod system_list;
mod system_panel;
mod system_rows;
mod transcription;

use crate::{
    components::{
        dialog::{Dialog, DialogTone},
        page::Page,
        toast::{Toast, ToastKind, use_toasts},
    },
    platform::IS_WEB,
    settings_store::use_settings,
};
use about::AboutSection;
use appearance::AppearanceSection;
use dioxus::prelude::*;
use logging::LoggingSection;
use performance::PerformanceSection;
use saved_tick::SavedTick;
use section_nav::SectionNav;
use transcription::TranscriptionSection;

/// The Settings page: jump links, the groups of settings, and the reset.
#[component]
pub fn Settings() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let mut reset_open = use_signal(|| false);

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
        Page {
            title: "Settings",
            actions: rsx! {
                SavedTick {}
            },
            SectionNav {}
            div { class: "settings",
                AppearanceSection {}
                TranscriptionSection {}
                if !IS_WEB {
                    PerformanceSection {}
                }
                LoggingSection {}
                AboutSection { on_reset: move |()| reset_open.set(true) }
            }
        }
        if reset_open() {
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
