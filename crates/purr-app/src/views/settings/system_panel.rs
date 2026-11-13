use super::{system_list::SystemList, system_rows::rows};
use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonVariant},
        icon::IconName,
    },
    utils::dom_id::use_dom_id,
};
use dioxus::prelude::*;
use purr_common::accel::SystemReport;

/// What this computer offers: the backends this build can use with their status, and the devices.
#[component]
pub fn SystemPanel(system: Resource<Result<SystemReport, String>>) -> Element {
    let heading = use_dom_id("system");
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
                        "“Built, unavailable”: this build supports the backend, but no device or driver responds."
                    }
                },
            }
        }
    }
}
