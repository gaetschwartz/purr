use super::{changes::report, run_on::RunOn, system_panel::SystemPanel};
use crate::{
    components::{
        group::{Group, Item},
        select::{Select, SelectOption},
        stepper::Stepper,
        switch::Switch,
        toast::use_toasts,
    },
    platform,
    settings_store::use_settings,
};
use dioxus::prelude::*;
use purr_common::platform::Platform as _;

const THREAD_LIMIT_WITHOUT_PROBE: u32 = 64;

/// Where transcription runs, this computer's devices and the processor threads; desktop only.
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
    let (run_on, fallback, threads) = settings.with(|settings| {
        let performance = &settings.performance;
        (
            RunOn::of(performance),
            performance.accel_fallback,
            performance.threads,
        )
    });
    let probed = (*system.read()).clone();
    let loaded = probed.as_ref().and_then(|result| result.as_ref().ok());
    let choices: Vec<SelectOption<RunOn>> = super::run_on::options(loaded, run_on)
        .into_iter()
        .map(|option| SelectOption::new(option.value, option.label).disabled(option.disabled))
        .collect();
    let max_threads = loaded
        .and_then(|report| report.cpu.logical_threads)
        .unwrap_or(THREAD_LIMIT_WITHOUT_PROBE)
        .max(1);

    rsx! {
        Group {
            Item {
                title: "Run on",
                control: rsx! {
                    Select {
                        label: "Run on",
                        value: run_on,
                        options: choices,
                        onchange: move |choice: RunOn| {
                            report(toasts, settings.update(|s| choice.apply(&mut s.performance)));
                        },
                    }
                },
            }
            if run_on.offers_fallback() {
                Item {
                    title: "Fall back if unavailable",
                    control: rsx! {
                        Switch {
                            label: "Fall back to the next best backend if unavailable",
                            checked: fallback,
                            onchange: move |value| report(toasts, settings.update(|s| s.performance.accel_fallback = value)),
                        }
                    },
                }
            }
        }
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
