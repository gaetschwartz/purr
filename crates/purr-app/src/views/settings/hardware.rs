use super::{changes::report, system_panel::SystemPanel};
use crate::{
    components::{
        group::{Group, Item},
        stepper::Stepper,
        toast::use_toasts,
    },
    platform,
    settings_store::use_settings,
};
use dioxus::prelude::*;
use purr_common::platform::Platform as _;

const THREAD_LIMIT_WITHOUT_PROBE: u32 = 64;

/// This computer's devices in the order they are tried, and the processor threads; desktop only.
#[component]
pub fn HardwarePage() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let system = use_resource(|| async {
        platform::get_platform()
            .system_report()
            .await
            .map_err(|error| error.to_string())
    });
    let threads = settings.with(|settings| settings.performance.threads);
    let probed = (*system.read()).clone();
    let loaded = probed.as_ref().and_then(|result| result.as_ref().ok());
    let max_threads = loaded
        .and_then(|report| report.cpu.logical_threads)
        .unwrap_or(THREAD_LIMIT_WITHOUT_PROBE)
        .max(1);

    rsx! {
        SystemPanel { system }
        Group {
            Item {
                title: "CPU threads",
                subtitle: "Auto uses all available threads.",
                control: rsx! {
                    Stepper {
                        label: "CPU threads",
                        value: threads.unwrap_or(0),
                        min: 0,
                        max: max_threads,
                        display: threads.is_none().then(|| "Auto".to_owned()),
                        onchange: move |value: u32| {
                            report(
                                toasts,
                                settings.update(|s| s.performance.threads = (value > 0).then_some(value)),
                            );
                        },
                    }
                },
            }
        }
    }
}
