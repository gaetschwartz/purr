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
use purr_common::accel::{AccelPolicy, AccelSelection};

/// Where browser transcriptions run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunOn {
    /// WebGPU when it works here, the CPU otherwise.
    Automatic,
    Cpu,
}

/// What this browser offers for transcription, and whether to keep it on the CPU.
#[component]
pub fn BrowserPanel() -> Element {
    let models = use_models();
    let settings = use_settings();
    let toasts = use_toasts();
    let heading = use_dom_id("browser");
    let rows = browser_rows(&models.gpu.read());
    let run_on =
        if settings.with(|settings| settings.performance.accel_policy()) == AccelPolicy::CPU_ONLY {
            RunOn::Cpu
        } else {
            RunOn::Automatic
        };

    rsx! {
        section { aria_labelledby: heading.clone(),
            h2 { class: "group-h", id: heading, "This browser" }
            SystemList { label: "WebGPU in this browser", rows }
            Group {
                Item {
                    title: "Run on",
                    stackable: true,
                    control: rsx! {
                        Segmented {
                            label: "Run on",
                            value: run_on,
                            segments: vec![
                                Segment::new(RunOn::Automatic, "Automatic"),
                                Segment::new(RunOn::Cpu, "CPU"),
                            ],
                            onchange: move |choice| {
                                report(
                                    toasts,
                                    settings
                                        .update(|s| {
                                            s.performance.gpu_acceleration = true;
                                            s.performance.accel_backend = match choice {
                                                RunOn::Automatic => AccelSelection::Auto,
                                                RunOn::Cpu => AccelSelection::Cpu,
                                            };
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
