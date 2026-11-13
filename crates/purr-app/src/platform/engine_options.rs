//! How the settings and the machine's probe become purr-core configuration and reports.

use purr_common::{
    accel::{
        AccelBackend, BackendReport, BackendStatus, BackendUnavailable, LastRun, PassedOver,
        PassedReason, SystemReport,
    },
    platform::{DeviceInfo, DeviceType, EngineOptions},
};
use purr_core::{
    TranscriptionConfig,
    accel::{
        backend::AccelBackend as CoreBackend,
        devices::{DetectedDevice, detect},
        inventory::Inventory,
        plan::backend_status,
        report::{AccelReport, AttemptOutcome, UnavailableReason},
    },
    cpu,
    dev::Device,
};
use std::collections::HashMap;

/// `config` run the way `engine` asks.
pub fn apply_engine(config: TranscriptionConfig, engine: &EngineOptions) -> TranscriptionConfig {
    let config = config
        .with_devices(engine.devices.clone())
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
        UnavailableReason::RuntimeLibraryMissing { library, searched } => {
            BackendUnavailable::RuntimeLibraryMissing {
                library,
                searched: searched
                    .iter()
                    .map(|dir| dir.display().to_string())
                    .collect(),
            }
        }
        UnavailableReason::NotBuiltOrNoDevice => BackendUnavailable::NotBuiltOrNoDevice,
    }
}

fn capabilities(device: &Device) -> Option<HashMap<String, serde_json::Value>> {
    device.caps.map(|caps| {
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
    })
}

/// A detected device in the platform's neutral description.
pub fn device_info(detected: &DetectedDevice, device: Option<Device>) -> DeviceInfo {
    match device {
        Some(device) => DeviceInfo {
            id: device.id,
            name: device.name.clone(),
            description: Some(device.description.clone()),
            device_type: device.tpe,
            memory_free: Some(device.vram_free),
            memory_total: Some(device.vram_total),
            capabilities: capabilities(&device),
            key: Some(detected.key.clone()),
        },
        None => DeviceInfo {
            id: -1,
            name: detected.name.clone(),
            description: None,
            device_type: DeviceType::Accel,
            memory_free: None,
            memory_total: None,
            capabilities: None,
            key: Some(detected.key.clone()),
        },
    }
}

/// What the last transcription ran on, from the report the core made while loading the model.
pub fn last_run(report: &AccelReport) -> LastRun {
    LastRun {
        backend: report.selected.into(),
        device: report.device.clone(),
        passed_over: report
            .attempts
            .iter()
            .map(|attempt| PassedOver {
                key: attempt.key.clone(),
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

/// Every backend in priority order with whether it can run and its devices, which come in the
/// order transcription tries them when nobody arranged them.
pub fn backend_reports(inventory: &Inventory) -> Vec<BackendReport> {
    let detected = detect(inventory);
    CoreBackend::DEFAULT_PRIORITY
        .iter()
        .map(|backend| BackendReport {
            backend: AccelBackend::from(*backend),
            status: match backend_status(*backend, inventory, &detected) {
                Ok(()) => BackendStatus::Available,
                Err(reason) => BackendStatus::Unavailable(unavailable(reason)),
            },
            devices: detected
                .iter()
                .filter(|device| device.backend == *backend)
                .map(|device| {
                    let found = device.ggml_id.and_then(|id| {
                        inventory
                            .devices
                            .iter()
                            .find(|candidate| candidate.id == id)
                            .cloned()
                    });
                    device_info(device, found)
                })
                .collect(),
        })
        .collect()
}

/// Probes this machine; this calls into the native backends and can take a moment.
pub fn probe(last_run: Option<LastRun>) -> SystemReport {
    let inventory = Inventory::detect();
    SystemReport {
        backends: backend_reports(&inventory),
        cpu: cpu::report(&inventory.devices),
        last_run,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use purr_common::devices::{DeviceKey, DeviceOrder};
    use purr_core::accel::{
        backend::BuildStatus,
        report::{AccelAttempt, AttemptFailure},
    };

    #[test]
    fn the_engine_options_reach_the_core_configuration() {
        let config = apply_engine(
            TranscriptionConfig::new(),
            &EngineOptions {
                devices: DeviceOrder::cpu_only(),
                threads: Some(5),
                temperature: 0.5,
            },
        );
        assert_eq!(config.devices, DeviceOrder::cpu_only());
        assert_eq!(config.num_threads, Some(5));
        assert!((config.temperature - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn without_threads_the_core_chooses() {
        let config = apply_engine(TranscriptionConfig::new(), &EngineOptions::default());
        assert_eq!(config.num_threads, None);
        assert_eq!(config.devices, DeviceOrder::automatic());
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
            runtime_dirs: vec![PathBuf::from("runtime")],
        }
    }

    fn device(id: i32, description: &str, tpe: DeviceType, backend: CoreBackend) -> Device {
        Device {
            id,
            name: format!("dev{id}"),
            description: description.to_owned(),
            device_id: None,
            vram_free: 0,
            vram_total: 0,
            tpe,
            backend: Some(backend),
            caps: None,
            vulkan_driver: None,
        }
    }

    fn cpu_device() -> Device {
        device(0, "A processor", DeviceType::Cpu, CoreBackend::Cpu)
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
                        library: "cudart64_13.dll".to_owned(),
                        searched: vec!["runtime".to_owned()],
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
    fn devices_keep_their_kind_their_key_and_the_order_they_are_tried_in() {
        let reports = backend_reports(&inventory(
            vec![
                device(
                    0,
                    "AMD Radeon(TM) Graphics",
                    DeviceType::Igpu,
                    CoreBackend::Vulkan,
                ),
                device(
                    1,
                    "NVIDIA GeForce RTX 2070 SUPER",
                    DeviceType::Gpu,
                    CoreBackend::Vulkan,
                ),
                cpu_device(),
            ],
            &[CoreBackend::Vulkan, CoreBackend::Cpu],
        ));
        let vulkan = reports
            .iter()
            .find(|report| report.backend == AccelBackend::Vulkan)
            .map(|report| {
                report
                    .devices
                    .iter()
                    .map(|device| (device.device_type, device.key.clone()))
                    .collect::<Vec<_>>()
            });
        assert_eq!(
            vulkan,
            Some(vec![
                (
                    DeviceType::Gpu,
                    Some(DeviceKey::new(
                        AccelBackend::Vulkan,
                        "NVIDIA GeForce RTX 2070 SUPER",
                        0
                    ))
                ),
                (
                    DeviceType::Igpu,
                    Some(DeviceKey::new(
                        AccelBackend::Vulkan,
                        "AMD Radeon(TM) Graphics",
                        0
                    ))
                ),
            ])
        );
    }

    #[test]
    fn the_last_run_names_the_backend_and_what_was_passed_over() {
        let vulkan = DeviceKey::new(AccelBackend::Vulkan, "AMD Radeon(TM) Graphics", 0);
        let report = AccelReport {
            selected: CoreBackend::Cpu,
            key: DeviceKey::cpu(),
            device: None,
            attempts: vec![AccelAttempt::new(
                vulkan.clone(),
                AttemptOutcome::Failed(AttemptFailure::ContextInit),
            )],
        };
        assert_eq!(
            last_run(&report),
            LastRun {
                backend: AccelBackend::Cpu,
                device: None,
                passed_over: vec![PassedOver {
                    key: vulkan,
                    reason: PassedReason::Failed(AttemptFailure::ContextInit.to_string()),
                }],
            }
        );
    }
}
