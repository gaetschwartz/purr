use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonSize},
        icon::IconName,
    },
    main_app::Route,
    transcribe::{engine::fallback_notice, session::RunRecord},
    utils::delay::sleep,
    views::settings::sections::SettingsSection,
};
use dioxus::prelude::*;
use std::time::Duration;

const ROUTE_SETTLE: Duration = Duration::from_millis(150);

fn jump_script(anchor: &str) -> String {
    format!("document.getElementById({anchor:?})?.scrollIntoView({{ block: 'start' }})")
}

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
                    onclick: move |_| {
                        navigator.push(Route::Settings {});
                        spawn(async move {
                            sleep(ROUTE_SETTLE).await;
                            document::eval(&jump_script(SettingsSection::Performance.anchor()));
                        });
                    },
                    "Open Performance settings"
                }
            },
            "{notice.detail}"
        }
    }
}
