//! How the settings and the machine's probe become purr-core configuration and reports.

use purr_common::{
    accel::{
        AccelBackend, AccelPolicy, BackendReport, BackendStatus, BackendUnavailable, CpuFeature,
        CpuReport, LastRun, PassedOver, PassedReason, SystemReport,
    },
    platform::{DeviceInfo, DeviceType, EngineOptions},
};
use purr_core::{
    TranscriptionConfig,
    accel::{
        backend::AccelBackend as CoreBackend,
        config::AccelConfig,
        inventory::Inventory,
        plan::locate,
        report::{AccelReport, AttemptOutcome, UnavailableReason},
    },
    dev::{self, Device, SystemCapabilities},
};
use std::collections::HashMap;

/// The purr-core name of `backend`.
pub const fn core_backend(backend: AccelBackend) -> CoreBackend {
    match backend {
        AccelBackend::Cuda => CoreBackend::Cuda,
        AccelBackend::Hip => CoreBackend::Hip,
        AccelBackend::CoreMl => CoreBackend::CoreMl,
        AccelBackend::Metal => CoreBackend::Metal,
        AccelBackend::Sycl => CoreBackend::Sycl,
        AccelBackend::Vulkan => CoreBackend::Vulkan,
        AccelBackend::Cpu => CoreBackend::Cpu,
    }
}

/// The settings name of the purr-core `backend`.
pub const fn backend_of_core(backend: CoreBackend) -> AccelBackend {
    match backend {
        CoreBackend::Cuda => AccelBackend::Cuda,
        CoreBackend::Hip => AccelBackend::Hip,
        CoreBackend::CoreMl => AccelBackend::CoreMl,
        CoreBackend::Metal => AccelBackend::Metal,
        CoreBackend::Sycl => AccelBackend::Sycl,
        CoreBackend::Vulkan => AccelBackend::Vulkan,
        CoreBackend::Cpu => AccelBackend::Cpu,
    }
}

/// The purr-core backend selection for `policy`.
pub fn accel_config(policy: AccelPolicy) -> AccelConfig {
    match policy {
        AccelPolicy::Auto => AccelConfig::default(),
        AccelPolicy::Forced {
            backend: AccelBackend::Cpu,
            ..
        } => AccelConfig::cpu_only(),
        AccelPolicy::Forced {
            backend,
            allow_fallback,
        } => AccelConfig::forced(core_backend(backend), allow_fallback),
    }
}

/// `config` run the way `engine` asks.
pub fn apply_engine(config: TranscriptionConfig, engine: &EngineOptions) -> TranscriptionConfig {
    let config = config
        .with_accel(accel_config(engine.accel))
        .with_temperature(engine.temperature as f32);
    match engine.threads {
        Some(threads) => config.with_threads(threads as usize),
        None => config,
    }
}

fn unavailable(reason: UnavailableReason) -> BackendUnavailable {
    match reason {
        UnavailableReason::NotBuilt => BackendUnavailable::NotBuilt,
        UnavailableReason::NoDevice => BackendUnavailable::NoDevice,
        UnavailableReason::RuntimeLibraryMissing { library } => {
            BackendUnavailable::RuntimeLibraryMissing { library }
        }
        UnavailableReason::NotBuiltOrNoDevice => BackendUnavailable::NotBuiltOrNoDevice,
    }
}

/// A device in the platform's neutral description.
pub fn device_info(device: Device) -> DeviceInfo {
    DeviceInfo {
        id: device.id,
        name: device.name,
        description: Some(device.description),
        device_type: match device.tpe {
            dev::DeviceType::Cpu => DeviceType::Cpu,
            dev::DeviceType::Gpu | dev::DeviceType::Igpu => DeviceType::Gpu,
            dev::DeviceType::Accel => DeviceType::Accel,
            dev::DeviceType::Unknown => DeviceType::Unknown,
        },
        memory_free: Some(device.vram_free),
        memory_total: Some(device.vram_total),
        capabilities: device.caps.map(|caps| {
            HashMap::from([
                ("async".to_owned(), serde_json::Value::Bool(caps.async_)),
                (
                    "host_buffer".to_owned(),
                    serde_json::Value::Bool(caps.host_buffer),
                ),
                (
                    "buffer_from_host_ptr".to_owned(),
                    serde_json::Value::Bool(caps.buffer_from_host_ptr),
                ),
                ("events".to_owned(), serde_json::Value::Bool(caps.events)),
            ])
        }),
    }
}

/// What the last transcription ran on, from the report the core made while loading the model.
pub fn last_run(report: &AccelReport) -> LastRun {
    LastRun {
        backend: backend_of_core(report.selected),
        device: report.device.clone(),
        passed_over: report
            .attempts
            .iter()
            .map(|attempt| PassedOver {
                backend: backend_of_core(attempt.backend),
                reason: match &attempt.outcome {
                    AttemptOutcome::Unavailable(reason) => {
                        PassedReason::Unavailable(unavailable(reason.clone()))
                    }
                    AttemptOutcome::Failed(failure) => PassedReason::Failed(failure.to_string()),
                },
            })
            .collect(),
    }
}

/// Every backend in priority order with whether it can run and its devices.
pub fn backend_reports(inventory: &Inventory) -> Vec<BackendReport> {
    CoreBackend::DEFAULT_PRIORITY
        .iter()
        .map(|backend| BackendReport {
            backend: backend_of_core(*backend),
            status: match locate(*backend, inventory) {
                Ok(_) => BackendStatus::Available,
                Err(reason) => BackendStatus::Unavailable(unavailable(reason)),
            },
            devices: inventory
                .devices
                .iter()
                .filter(|device| device.backend == Some(*backend))
                .cloned()
                .map(device_info)
                .collect(),
        })
        .collect()
}

fn cpu_name(devices: &[Device]) -> Option<String> {
    devices
        .iter()
        .find(|device| device.tpe == dev::DeviceType::Cpu)
        .map(|device| device.description.trim())
        .filter(|name| !name.is_empty() && !name.eq_ignore_ascii_case("cpu"))
        .map(str::to_owned)
}

/// The processor's name, instruction sets and thread count.
pub fn cpu_report(capabilities: &SystemCapabilities, devices: &[Device]) -> CpuReport {
    let features = [
        (capabilities.avx, CpuFeature::Avx),
        (capabilities.avx2, CpuFeature::Avx2),
        (capabilities.fma, CpuFeature::Fma),
        (capabilities.f16c, CpuFeature::F16c),
    ]
    .into_iter()
    .filter_map(|(present, feature)| present.then_some(feature))
    .collect();
    CpuReport {
        name: cpu_name(devices),
        logical_threads: std::thread::available_parallelism()
            .ok()
            .and_then(|threads| u32::try_from(threads.get()).ok()),
        features,
    }
}

/// Probes this machine; this calls into the native backends and can take a moment.
pub fn probe(last_run: Option<LastRun>) -> SystemReport {
    let inventory = Inventory::detect();
    SystemReport {
        backends: backend_reports(&inventory),
        cpu: cpu_report(&SystemCapabilities::get(), &inventory.devices),
        last_run,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_core::accel::{
        backend::BuildStatus,
        report::{AccelAttempt, AttemptFailure, AttemptOutcome},
    };

    #[test]
    fn the_priority_order_and_the_backends_match_purr_core() {
        let core: Vec<_> = AccelBackend::PRIORITY
            .iter()
            .map(|b| core_backend(*b))
            .collect();
        assert_eq!(core, CoreBackend::DEFAULT_PRIORITY);
        for backend in CoreBackend::ALL {
            assert_eq!(core_backend(backend_of_core(*backend)), *backend);
        }
    }

    #[test]
    fn automatic_policy_is_the_default_priority_list() {
        assert_eq!(accel_config(AccelPolicy::Auto), AccelConfig::default());
    }

    #[test]
    fn cpu_only_policy_is_a_forced_cpu_without_fallback() {
        assert_eq!(accel_config(AccelPolicy::CPU_ONLY), AccelConfig::cpu_only());
        assert_eq!(
            accel_config(AccelPolicy::Forced {
                backend: AccelBackend::Cpu,
                allow_fallback: true
            }),
            AccelConfig::cpu_only()
        );
    }

    #[test]
    fn a_forced_backend_keeps_the_fallback_choice() {
        for allow_fallback in [true, false] {
            assert_eq!(
                accel_config(AccelPolicy::Forced {
                    backend: AccelBackend::Vulkan,
                    allow_fallback
                }),
                AccelConfig::forced(CoreBackend::Vulkan, allow_fallback)
            );
        }
    }

    #[test]
    fn the_engine_options_reach_the_core_configuration() {
        let config = apply_engine(
            TranscriptionConfig::new(),
            &EngineOptions {
                accel: AccelPolicy::CPU_ONLY,
                threads: Some(5),
                temperature: 0.5,
            },
        );
        assert_eq!(config.accel, AccelConfig::cpu_only());
        assert_eq!(config.num_threads, Some(5));
        assert!((config.temperature - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn without_threads_the_core_chooses() {
        let config = apply_engine(TranscriptionConfig::new(), &EngineOptions::default());
        assert_eq!(config.num_threads, None);
        assert_eq!(config.accel, AccelConfig::default());
    }

    fn inventory(devices: Vec<Device>, built: &[CoreBackend]) -> Inventory {
        Inventory {
            devices,
            build: CoreBackend::ALL
                .iter()
                .map(|backend| {
                    let status = match (built.contains(backend), backend) {
                        (true, _) => BuildStatus::Built,
                        (false, CoreBackend::Hip | CoreBackend::Sycl) => BuildStatus::Unknown,
                        (
                            false,
                            CoreBackend::Cuda
                            | CoreBackend::CoreMl
                            | CoreBackend::Metal
                            | CoreBackend::Vulkan
                            | CoreBackend::Cpu,
                        ) => BuildStatus::NotBuilt,
                    };
                    (*backend, status)
                })
                .collect(),
            missing_runtime: HashMap::from([(CoreBackend::Cuda, "cudart64_13.dll".to_owned())]),
        }
    }

    fn cpu_device() -> Device {
        Device {
            id: 0,
            name: "CPU".to_owned(),
            description: "A processor".to_owned(),
            vram_free: 0,
            vram_total: 0,
            tpe: dev::DeviceType::Cpu,
            backend: Some(CoreBackend::Cpu),
            caps: None,
        }
    }

    #[test]
    fn backends_are_reported_in_priority_order_with_the_reason_they_cannot_run() {
        let reports = backend_reports(&inventory(
            vec![cpu_device()],
            &[CoreBackend::Cuda, CoreBackend::Vulkan, CoreBackend::Cpu],
        ));
        let summary: Vec<_> = reports
            .iter()
            .map(|report| (report.backend, report.status.clone(), report.devices.len()))
            .collect();
        assert_eq!(
            summary,
            [
                (
                    AccelBackend::Cuda,
                    BackendStatus::Unavailable(BackendUnavailable::RuntimeLibraryMissing {
                        library: "cudart64_13.dll".to_owned()
                    }),
                    0
                ),
                (
                    AccelBackend::Hip,
                    BackendStatus::Unavailable(BackendUnavailable::NotBuiltOrNoDevice),
                    0
                ),
                (
                    AccelBackend::CoreMl,
                    BackendStatus::Unavailable(BackendUnavailable::NotBuilt),
                    0
                ),
                (
                    AccelBackend::Metal,
                    BackendStatus::Unavailable(BackendUnavailable::NotBuilt),
                    0
                ),
                (
                    AccelBackend::Sycl,
                    BackendStatus::Unavailable(BackendUnavailable::NotBuiltOrNoDevice),
                    0
                ),
                (
                    AccelBackend::Vulkan,
                    BackendStatus::Unavailable(BackendUnavailable::NoDevice),
                    0
                ),
                (AccelBackend::Cpu, BackendStatus::Available, 1),
            ]
        );
    }

    #[test]
    fn the_last_run_names_the_backend_and_what_was_passed_over() {
        let report = AccelReport {
            selected: CoreBackend::Cpu,
            device: Some("CPU".to_owned()),
            attempts: vec![AccelAttempt {
                backend: CoreBackend::Vulkan,
                outcome: AttemptOutcome::Failed(AttemptFailure::ContextInit),
            }],
        };
        assert_eq!(
            last_run(&report),
            LastRun {
                backend: AccelBackend::Cpu,
                device: Some("CPU".to_owned()),
                passed_over: vec![PassedOver {
                    backend: AccelBackend::Vulkan,
                    reason: PassedReason::Failed(AttemptFailure::ContextInit.to_string()),
                }],
            }
        );
    }

    #[test]
    fn instruction_sets_are_listed_in_a_stable_order() {
        let report = cpu_report(
            &SystemCapabilities {
                avx: true,
                avx2: true,
                fma: false,
                f16c: true,
            },
            &[],
        );
        assert_eq!(
            report.features,
            [CpuFeature::Avx, CpuFeature::Avx2, CpuFeature::F16c]
        );
    }

    fn named_cpu(description: &str) -> Device {
        Device {
            description: description.to_owned(),
            ..cpu_device()
        }
    }

    #[test]
    fn the_processor_is_named_by_its_device_description() {
        assert_eq!(
            cpu_name(&[named_cpu("  AMD Ryzen 9 7950X 16-Core Processor ")]),
            Some("AMD Ryzen 9 7950X 16-Core Processor".to_owned())
        );
    }

    #[test]
    fn a_processor_the_system_does_not_name_stays_unnamed() {
        assert_eq!(cpu_name(&[named_cpu("CPU")]), None);
        assert_eq!(cpu_name(&[named_cpu("  ")]), None);
        assert_eq!(cpu_name(&[]), None);
    }
}
