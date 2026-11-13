use super::system_rows::device_name;
use purr_common::{
    accel::{
        AccelBackend, AccelSelection, BackendReport, BackendStatus, BackendUnavailable,
        SystemReport,
    },
    settings::PerformanceSettings,
};

/// Where transcription runs, as the Hardware page offers it.
///
/// It reads and writes the acceleration switch and the backend of [`PerformanceSettings`]; the
/// stored fields stay as they are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunOn {
    /// The best backend that works.
    Automatic,
    /// The processor only.
    Cpu,
    /// One accelerator; never [`AccelBackend::Cpu`], which is [`Self::Cpu`].
    Backend(AccelBackend),
}

impl RunOn {
    /// The choice the stored settings amount to.
    #[must_use]
    pub const fn of(performance: &PerformanceSettings) -> Self {
        if !performance.gpu_acceleration {
            return Self::Cpu;
        }
        match performance.accel_backend.backend() {
            None => Self::Automatic,
            Some(backend) => Self::named(backend),
        }
    }

    const fn named(backend: AccelBackend) -> Self {
        match backend {
            AccelBackend::Cpu => Self::Cpu,
            AccelBackend::Cuda
            | AccelBackend::Hip
            | AccelBackend::CoreMl
            | AccelBackend::Metal
            | AccelBackend::Sycl
            | AccelBackend::Vulkan => Self::Backend(backend),
        }
    }

    /// Stores the choice; the fallback switch is left alone.
    pub const fn apply(self, performance: &mut PerformanceSettings) {
        performance.gpu_acceleration = true;
        performance.accel_backend = match self {
            Self::Automatic => AccelSelection::Auto,
            Self::Cpu => AccelSelection::Cpu,
            Self::Backend(backend) => AccelSelection::of(backend),
        };
    }

    /// Whether the user can say what happens when the choice cannot run.
    #[must_use]
    pub const fn offers_fallback(self) -> bool {
        match self {
            Self::Backend(_) => true,
            Self::Automatic | Self::Cpu => false,
        }
    }
}

/// One entry of the "Run on" menu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOnOption {
    pub value: RunOn,
    pub label: String,
    pub disabled: bool,
}

impl RunOnOption {
    fn enabled(value: RunOn, label: impl Into<String>) -> Self {
        Self {
            value,
            label: label.into(),
            disabled: false,
        }
    }
}

fn devices_label(entry: &BackendReport) -> String {
    let backend = entry.backend.label();
    match entry.devices.as_slice() {
        [] => backend.to_owned(),
        [device] => format!("{backend} · {}", device_name(device)),
        devices => format!("{backend} · {} devices", devices.len()),
    }
}

fn backend_option(entry: &BackendReport) -> Option<RunOnOption> {
    let value = RunOn::named(entry.backend);
    match (&entry.status, value) {
        (_, RunOn::Cpu | RunOn::Automatic) => None,
        (BackendStatus::Available, _) => Some(RunOnOption::enabled(value, devices_label(entry))),
        (
            BackendStatus::Unavailable(
                BackendUnavailable::NoDevice | BackendUnavailable::RuntimeLibraryMissing { .. },
            ),
            _,
        ) => Some(RunOnOption {
            value,
            label: format!("{} (unavailable)", entry.backend.label()),
            disabled: true,
        }),
        (
            BackendStatus::Unavailable(
                BackendUnavailable::NotBuilt | BackendUnavailable::NotBuiltOrNoDevice,
            ),
            _,
        ) => None,
    }
}

/// The "Run on" menu: Automatic, the accelerators this build has, then the processor only.
///
/// `current` is always listed, so the menu never shows a choice the settings do not hold.
#[must_use]
pub fn options(report: Option<&SystemReport>, current: RunOn) -> Vec<RunOnOption> {
    let mut entries = vec![RunOnOption::enabled(
        RunOn::Automatic,
        "Automatic (best available)",
    )];
    if let Some(report) = report {
        entries.extend(report.backends.iter().filter_map(backend_option));
    }
    if let RunOn::Backend(backend) = current
        && !entries.iter().any(|entry| entry.value == current)
    {
        let label = if report.is_some() {
            format!("{} (unavailable)", backend.label())
        } else {
            backend.label().to_owned()
        };
        entries.push(RunOnOption::enabled(current, label));
    }
    entries.push(RunOnOption::enabled(RunOn::Cpu, "CPU only"));
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::{
        accel::{AccelPolicy, CpuReport},
        platform::{DeviceInfo, DeviceType},
    };

    fn stored(gpu: bool, backend: AccelSelection, fallback: bool) -> PerformanceSettings {
        PerformanceSettings {
            gpu_acceleration: gpu,
            accel_backend: backend,
            accel_fallback: fallback,
            threads: None,
        }
    }

    #[test]
    fn the_stored_fields_read_as_a_choice() {
        assert_eq!(
            RunOn::of(&stored(true, AccelSelection::Auto, true)),
            RunOn::Automatic
        );
        assert_eq!(
            RunOn::of(&stored(true, AccelSelection::Cpu, true)),
            RunOn::Cpu
        );
        assert_eq!(
            RunOn::of(&stored(true, AccelSelection::Vulkan, false)),
            RunOn::Backend(AccelBackend::Vulkan)
        );
    }

    #[test]
    fn acceleration_switched_off_reads_as_the_processor_only() {
        for backend in AccelSelection::ALL {
            assert_eq!(RunOn::of(&stored(false, backend, true)), RunOn::Cpu);
        }
    }

    #[test]
    fn a_choice_is_stored_in_the_existing_fields() {
        let store = |choice: RunOn| {
            let mut performance = stored(false, AccelSelection::Metal, false);
            choice.apply(&mut performance);
            performance
        };
        assert_eq!(
            store(RunOn::Automatic),
            stored(true, AccelSelection::Auto, false)
        );
        assert_eq!(store(RunOn::Cpu), stored(true, AccelSelection::Cpu, false));
        assert_eq!(
            store(RunOn::Backend(AccelBackend::Cuda)),
            stored(true, AccelSelection::Cuda, false)
        );
    }

    fn choices() -> Vec<RunOn> {
        let mut all = vec![RunOn::Automatic, RunOn::Cpu];
        all.extend(
            AccelBackend::PRIORITY
                .into_iter()
                .filter(|backend| *backend != AccelBackend::Cpu)
                .map(RunOn::Backend),
        );
        all
    }

    #[test]
    fn a_stored_choice_survives_the_round_trip() {
        for choice in choices() {
            let mut performance = PerformanceSettings::default();
            choice.apply(&mut performance);
            assert_eq!(RunOn::of(&performance), choice);
        }
    }

    #[test]
    fn a_choice_makes_the_transcription_pick_what_it_names() {
        for fallback in [true, false] {
            let policy = |choice: RunOn| {
                let mut performance = stored(true, AccelSelection::Auto, fallback);
                choice.apply(&mut performance);
                performance.accel_policy()
            };
            assert_eq!(policy(RunOn::Automatic), AccelPolicy::Auto);
            assert_eq!(policy(RunOn::Cpu), AccelPolicy::CPU_ONLY);
            assert_eq!(
                policy(RunOn::Backend(AccelBackend::Vulkan)),
                AccelPolicy::Forced {
                    backend: AccelBackend::Vulkan,
                    allow_fallback: fallback
                }
            );
        }
    }

    #[test]
    fn reading_the_settings_back_keeps_their_policy() {
        for gpu in [true, false] {
            for backend in AccelSelection::ALL {
                for fallback in [true, false] {
                    let before = stored(gpu, backend, fallback);
                    let mut after = before.clone();
                    RunOn::of(&before).apply(&mut after);
                    assert_eq!(after.accel_policy(), before.accel_policy());
                }
            }
        }
    }

    #[test]
    fn only_a_named_accelerator_offers_a_fallback() {
        assert!(!RunOn::Automatic.offers_fallback());
        assert!(!RunOn::Cpu.offers_fallback());
        assert!(RunOn::Backend(AccelBackend::Hip).offers_fallback());
    }

    fn device(description: &str) -> DeviceInfo {
        DeviceInfo {
            id: 0,
            name: "Vulkan0".to_owned(),
            description: Some(description.to_owned()),
            device_type: DeviceType::Gpu,
            memory_free: None,
            memory_total: None,
            capabilities: None,
        }
    }

    fn entry(
        backend: AccelBackend,
        status: BackendStatus,
        devices: Vec<DeviceInfo>,
    ) -> BackendReport {
        BackendReport {
            backend,
            status,
            devices,
        }
    }

    fn report() -> SystemReport {
        SystemReport {
            backends: vec![
                entry(
                    AccelBackend::Cuda,
                    BackendStatus::Unavailable(BackendUnavailable::NoDevice),
                    vec![],
                ),
                entry(
                    AccelBackend::Metal,
                    BackendStatus::Unavailable(BackendUnavailable::NotBuilt),
                    vec![],
                ),
                entry(
                    AccelBackend::Sycl,
                    BackendStatus::Unavailable(BackendUnavailable::NotBuiltOrNoDevice),
                    vec![],
                ),
                entry(
                    AccelBackend::Vulkan,
                    BackendStatus::Available,
                    vec![device("NVIDIA GeForce RTX 4070")],
                ),
                entry(
                    AccelBackend::Hip,
                    BackendStatus::Available,
                    vec![device("First"), device("Second")],
                ),
                entry(
                    AccelBackend::Cpu,
                    BackendStatus::Available,
                    vec![device("Some Processor 9000")],
                ),
            ],
            cpu: CpuReport {
                name: None,
                logical_threads: None,
                built_with: vec![],
                detected: None,
            },
            last_run: None,
        }
    }

    fn option(value: RunOn, label: &str, disabled: bool) -> RunOnOption {
        RunOnOption {
            value,
            label: label.to_owned(),
            disabled,
        }
    }

    #[test]
    fn the_menu_lists_automatic_the_built_accelerators_then_the_processor_only() {
        assert_eq!(
            options(Some(&report()), RunOn::Automatic),
            [
                option(RunOn::Automatic, "Automatic (best available)", false),
                option(
                    RunOn::Backend(AccelBackend::Cuda),
                    "CUDA (unavailable)",
                    true
                ),
                option(
                    RunOn::Backend(AccelBackend::Vulkan),
                    "Vulkan · NVIDIA GeForce RTX 4070",
                    false
                ),
                option(RunOn::Backend(AccelBackend::Hip), "ROCm · 2 devices", false),
                option(RunOn::Cpu, "CPU only", false),
            ]
        );
    }

    #[test]
    fn the_menu_before_the_probe_still_holds_the_stored_choice() {
        assert_eq!(
            options(None, RunOn::Backend(AccelBackend::Vulkan)),
            [
                option(RunOn::Automatic, "Automatic (best available)", false),
                option(RunOn::Backend(AccelBackend::Vulkan), "Vulkan", false),
                option(RunOn::Cpu, "CPU only", false),
            ]
        );
    }

    #[test]
    fn a_stored_accelerator_the_build_lacks_stays_in_the_menu() {
        let listed = options(Some(&report()), RunOn::Backend(AccelBackend::Metal));
        assert!(
            listed
                .iter()
                .any(|entry| entry.value == RunOn::Backend(AccelBackend::Metal))
        );
    }
}
