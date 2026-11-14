use super::{changes::report, device_list::DeviceList, engine_notes::engine_notes};
use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonVariant},
        icon::IconName,
        info_tip::InfoTip,
        toast::use_toasts,
    },
    settings_store::use_settings,
    transcribe::device_rows::{device_rows, moved, saved_list, toggled},
    utils::dom_id::use_dom_id,
};
use dioxus::prelude::*;
use purr_common::accel::SystemReport;

/// The devices of this computer in the order they are tried, to switch on and off and to move.
#[component]
pub fn SystemPanel(system: Resource<Result<SystemReport, String>>) -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let heading = use_dom_id("system");
    let state = (*system.read()).clone();

    rsx! {
        section { aria_labelledby: heading.clone(),
            div { class: "hrow",
                h2 { class: "group-h", id: heading, "Devices" }
                InfoTip {
                    about: "Devices",
                    text: "Tried from the top; the CPU is the fallback.",
                }
            }
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
                Some(Ok(found)) => {
                    let rows = device_rows(&settings.with(|settings| settings.performance.clone()), &found);
                    let list = saved_list(&rows);
                    let list_for_toggle = list.clone();
                    let notes = engine_notes(&found.engines);
                    rsx! {
                        for note in notes {
                            Banner {
                                key: "{note.engine.label()}",
                                tone: BannerTone::Warn,
                                title: note.headline(),
                                role: "status",
                                style: "margin-bottom:12px",
                                details: note.detail.map(|detail| rsx! { "{detail}" }),
                            }
                        }
                        DeviceList {
                            rows,
                            engines: found.engines.clone(),
                            on_move: move |(key, direction)| {
                                let next = moved(&list, &key, direction);
                                report(toasts, settings.update(|s| s.performance.set_devices(next)));
                            },
                            on_toggle: move |(key, enabled)| {
                                let next = toggled(&list_for_toggle, &key, enabled);
                                report(toasts, settings.update(|s| s.performance.set_devices(next)));
                            },
                        }
                    }
                }
            }
        }
    }
}
