use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonSize},
        icon::IconName,
    },
    transcribe::{engine::fallback_notice, session::RunRecord},
    views::settings::{jump::open_section, sections::SettingsSection},
};
use dioxus::prelude::*;

/// Says, without blocking the run, that it used other hardware than the one asked for.
#[component]
pub fn EngineNotice(record: RunRecord) -> Element {
    let navigator = use_navigator();
    let Some(notice) = record
        .run
        .engine
        .as_ref()
        .and_then(|ran| fallback_notice(record.accel, ran))
    else {
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
                    onclick: move |_| open_section(navigator, SettingsSection::Performance),
                    "Open Performance settings"
                }
            },
            "{notice.detail}"
        }
    }
}
