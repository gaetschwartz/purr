use super::{
    system_list::SystemList,
    system_rows::{last_run_note, rows},
};
use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonVariant},
        icon::IconName,
    },
    settings_store::use_settings,
    utils::dom_id::use_dom_id,
};
use dioxus::prelude::*;
use purr_common::accel::SystemReport;

/// What this computer offers: every backend with its status, the devices, and where the last
/// transcription ran.
#[component]
pub fn SystemPanel(system: Resource<Result<SystemReport, String>>) -> Element {
    let heading = use_dom_id("system");
    let settings = use_settings();
    let requested = settings.with(|settings| {
        settings
            .performance
            .gpu_acceleration
            .then(|| settings.performance.accel_backend.backend())
            .flatten()
    });
    let state = (*system.read()).clone();

    rsx! {
        section { aria_labelledby: heading.clone(),
            h2 { class: "group-h", id: heading, "This computer" }
            match state {
                None => rsx! {
                    p { class: "group-f", role: "status", style: "margin:0 16px", "Checking this computer…" }
                },
                Some(Err(error)) => rsx! {
                    Banner {
                        tone: BannerTone::Bad,
                        title: "Couldn't read the system information",
                        role: "alert",
                        actions: rsx! {
                            Button {
                                variant: ButtonVariant::Primary,
                                icon: IconName::Retry,
                                onclick: move |_| system.restart(),
                                "Try again"
                            }
                        },
                        "{error}"
                    }
                },
                Some(Ok(report)) => rsx! {
                    SystemList { label: "Detected backends and devices", rows: rows(&report) }
                    p { class: "group-f",
                        "Read-only. “Built, unavailable” means this copy of Purr supports the backend but the device or driver doesn't respond."
                    }
                    if let Some(note) = last_run_note(report.last_run.as_ref(), requested) {
                        div { style: "margin-top:14px",
                            if note.detail.is_empty() {
                                Banner {
                                    tone: BannerTone::Info,
                                    icon: IconName::Bolt,
                                    title: note.title,
                                    role: "status",
                                }
                            } else {
                                Banner {
                                    tone: if note.fell_back { BannerTone::Warn } else { BannerTone::Info },
                                    icon: IconName::Bolt,
                                    title: note.title,
                                    role: "status",
                                    "{note.detail}"
                                }
                            }
                        }
                    }
                },
            }
        }
    }
}
