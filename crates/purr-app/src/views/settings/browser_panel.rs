use super::{changes::report, system_list::SystemList, system_rows::browser_rows};
use crate::{
    components::{
        group::{Group, Item},
        segmented::{Segment, Segmented},
        toast::use_toasts,
    },
    hooks::use_models::use_models,
    settings_store::use_settings,
    utils::dom_id::use_dom_id,
};
use dioxus::prelude::*;
use purr_common::devices::{DeviceKey, DevicePreference};

/// Where browser transcriptions run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunOn {
    /// The browser's GPU, with the CPU as the way out when WebGPU cannot start.
    WebGpu,
    Cpu,
}

/// What this browser offers for transcription, and whether to run on its GPU or its CPU.
#[component]
pub fn BrowserPanel() -> Element {
    let models = use_models();
    let settings = use_settings();
    let toasts = use_toasts();
    let heading = use_dom_id("browser");
    let rows = browser_rows(&models.gpu.read());
    let run_on = if settings.with(|settings| settings.performance.device_order().is_cpu_only()) {
        RunOn::Cpu
    } else {
        RunOn::WebGpu
    };

    rsx! {
        section { aria_labelledby: heading.clone(),
            h2 { class: "group-h", id: heading, "This browser" }
            SystemList { label: "WebGPU in this browser", rows }
            Group {
                Item {
                    title: "Run on",
                    subtitle: "WebGPU uses this browser's GPU. If it cannot start, Purr uses the slower CPU.",
                    stackable: true,
                    control: rsx! {
                        Segmented {
                            label: "Run on",
                            value: run_on,
                            segments: vec![
                                Segment::new(RunOn::WebGpu, "WebGPU"),
                                Segment::new(RunOn::Cpu, "CPU"),
                            ],
                            onchange: move |choice| {
                                report(
                                    toasts,
                                    settings
                                        .update(|s| {
                                            s.performance.set_devices(match choice {
                                                RunOn::WebGpu => Vec::new(),
                                                RunOn::Cpu => {
                                                    vec![DevicePreference::enabled(DeviceKey::cpu())]
                                                }
                                            });
                                        }),
                                );
                            },
                        }
                    },
                }
            }
        }
    }
}
