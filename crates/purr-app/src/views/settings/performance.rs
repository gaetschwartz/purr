use super::{
    changes::report, sections::SettingsSection, system_panel::SystemPanel,
    system_rows::backend_options,
};
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
use purr_common::{accel::AccelSelection, platform::Platform as _};

const THREAD_LIMIT_WITHOUT_PROBE: u32 = 64;

/// Acceleration, the backend and threads; desktop only.
#[component]
pub fn PerformanceSection() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let system = use_resource(|| async {
        platform::get_platform()
            .system_report()
            .await
            .map_err(|error| error.to_string())
    });
    let (gpu, backend, fallback, threads) = settings.with(|settings| {
        let performance = &settings.performance;
        (
            performance.gpu_acceleration,
            performance.accel_backend,
            performance.accel_fallback,
            performance.threads,
        )
    });
    let probed = (*system.read()).clone();
    let loaded = probed.as_ref().and_then(|result| result.as_ref().ok());
    let backends: Vec<SelectOption<AccelSelection>> = backend_options(loaded)
        .into_iter()
        .map(|(selection, label)| SelectOption::new(selection, label))
        .collect();
    let max_threads = loaded
        .and_then(|report| report.cpu.logical_threads)
        .unwrap_or(THREAD_LIMIT_WITHOUT_PROBE)
        .max(1);
    let automatic = backend == AccelSelection::Auto;
    let fallback_hint = automatic.then_some("Always on with Automatic.");

    rsx! {
        div { class: "sect", id: SettingsSection::Performance.anchor(),
            Group { header: SettingsSection::Performance.label(),
                Item {
                    title: "GPU acceleration",
                    control: rsx! {
                        Switch {
                            label: "GPU acceleration",
                            checked: gpu,
                            onchange: move |value| report(toasts, settings.update(|s| s.performance.gpu_acceleration = value)),
                        }
                    },
                }
                Item {
                    title: "Backend",
                    control: rsx! {
                        Select {
                            label: "Acceleration backend",
                            value: backend,
                            options: backends,
                            disabled: !gpu,
                            onchange: move |choice| report(toasts, settings.update(|s| s.performance.accel_backend = choice)),
                        }
                    },
                }
                Item {
                    title: "Fall back if unavailable",
                    subtitle: fallback_hint.map(str::to_owned),
                    control: rsx! {
                        Switch {
                            label: "Fall back to the next best backend if unavailable",
                            checked: fallback || automatic,
                            disabled: !gpu || automatic,
                            onchange: move |value| report(toasts, settings.update(|s| s.performance.accel_fallback = value)),
                        }
                    },
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
}
