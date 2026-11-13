use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonSize},
        icon::IconName,
    },
    transcribe::{engine::fallback_notice, session::RunRecord},
    views::settings::{dialog_handle::use_settings_dialog, settings_page::SettingsPage},
};
use dioxus::prelude::*;
use purr_common::browser_engine::BrowserEngine;

/// Says, without blocking the run, that it used other hardware than the one asked for.
#[component]
pub fn EngineNotice(record: RunRecord) -> Element {
    let dialog = use_settings_dialog();
    if let Some(BrowserEngine::Cpu(reason)) = record.run.browser_engine {
        return rsx! {
            if let Some(notice) = reason.notice() {
                p { class: "foot", role: "status", id: "engine-notice", style: "margin:0",
                    "{notice}"
                }
            }
        };
    }
    let Some(notice) = record.run.engine.as_ref().and_then(fallback_notice) else {
        return rsx! {};
    };

    rsx! {
        Banner {
            tone: BannerTone::Warn,
            icon: IconName::Bolt,
            title: notice.title,
            role: "status",
            id: "engine-notice",
            actions: rsx! {
                Button {
                    size: ButtonSize::Small,
                    onclick: move |_| dialog.open(SettingsPage::Hardware),
                    "Open Hardware settings"
                }
            },
            "{notice.detail}"
        }
    }
}
