use crate::{
    components::icon::{Icon, IconName, IconSize},
    settings_store::{status::SaveStatus, use_settings},
    utils::delay::sleep,
};
use dioxus::prelude::*;
use std::time::Duration;

const TICK_VISIBLE: Duration = Duration::from_millis(1600);

/// The quiet "Saved" in the page header: it shows for a moment after each saved change, and a
/// failed or paused save stays in view with its reason.
#[component]
pub fn SavedTick() -> Element {
    let settings = use_settings();
    let saved_at = use_memo(move || match settings.status() {
        SaveStatus::Saved { at_millis } => Some(at_millis),
        SaveStatus::Idle | SaveStatus::Saving | SaveStatus::Failed { .. } | SaveStatus::Paused => {
            None
        }
    });
    let opened_on = use_hook(|| *saved_at.peek());
    let mut visible = use_signal(|| false);
    use_resource(move || async move {
        if saved_at().is_some() && saved_at() != opened_on {
            visible.set(true);
            sleep(TICK_VISIBLE).await;
            visible.set(false);
        }
    });

    let problem = match settings.status() {
        SaveStatus::Failed { .. } => Some("Not saved"),
        SaveStatus::Paused => Some("Not saved until the settings are reset"),
        SaveStatus::Idle | SaveStatus::Saving | SaveStatus::Saved { .. } => None,
    };
    if let Some(problem) = problem {
        return rsx! {
            span { class: "saved on warn", role: "status",
                Icon { name: IconName::Warning, size: IconSize::Small }
                "{problem}"
            }
        };
    }
    rsx! {
        span { class: if visible() { "saved on" } else { "saved" }, aria_hidden: "true",
            Icon { name: IconName::CheckCircle, size: IconSize::Small }
            "Saved"
        }
        span { class: "sr", role: "status",
            if visible() {
                "Saved"
            }
        }
    }
}
