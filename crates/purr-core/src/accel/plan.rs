use crate::{
    dev::{Device, DeviceType},
    error::AccelerationError,
};

use super::{
    backend::{AccelBackend, BuildStatus},
    config::{AccelConfig, AccelMode},
    inventory::Inventory,
    report::{AccelAttempt, AttemptOutcome, UnavailableReason},
};

/// A device whisper.cpp can be pointed at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuTarget {
    /// Index of the device in the global ggml device list.
    pub device_id: i32,
    /// Value for whisper's `gpu_device`: whisper counts only GPU and iGPU devices, in global
    /// order, so this differs from `device_id` as soon as another kind of device comes first.
    pub ordinal: i32,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    Cpu,
    CoreMl,
    Gpu(GpuTarget),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub backend: AccelBackend,
    pub target: Target,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Plan {
    /// Usable candidates, in the order they should be tried.
    pub candidates: Vec<Candidate>,
    /// Backends of the attempt order that cannot be used at all.
    pub skipped: Vec<AccelAttempt>,
}

fn is_gpu(device: &Device) -> bool {
    matches!(device.tpe, DeviceType::Gpu | DeviceType::Igpu)
}

/// Where `backend` would run, or why it cannot.
pub fn locate(
    backend: AccelBackend,
    inventory: &Inventory,
) -> Result<Candidate, UnavailableReason> {
    let unavailable = || match inventory.build_status(backend) {
        BuildStatus::Built => UnavailableReason::NoDevice,
        BuildStatus::NotBuilt => UnavailableReason::NotBuilt,
        BuildStatus::Unknown => UnavailableReason::NotBuiltOrNoDevice,
    };
    if let Some(library) = inventory.missing_runtime.get(&backend) {
        return Err(UnavailableReason::RuntimeLibraryMissing {
            library: library.clone(),
            searched: inventory.runtime_dirs.clone(),
        });
    }
    let target = match backend {
        AccelBackend::Cpu => Target::Cpu,
        AccelBackend::CoreMl => match inventory.build_status(backend) {
            BuildStatus::Built => Target::CoreMl,
            BuildStatus::NotBuilt | BuildStatus::Unknown => return Err(unavailable()),
        },
        AccelBackend::Cuda
        | AccelBackend::Hip
        | AccelBackend::Metal
        | AccelBackend::Sycl
        | AccelBackend::Vulkan => {
            let device = inventory
                .devices
                .iter()
                .find(|d| d.backend == Some(backend) && is_gpu(d))
                .ok_or_else(unavailable)?;
            let ordinal = inventory
                .devices
                .iter()
                .filter(|d| is_gpu(d))
                .position(|d| d.id == device.id)
                .and_then(|ordinal| i32::try_from(ordinal).ok())
                .ok_or_else(unavailable)?;
            Target::Gpu(GpuTarget {
                device_id: device.id,
                ordinal,
                name: device.name.clone(),
            })
        }
    };
    Ok(Candidate { backend, target })
}

/// Decide which backends to try, without touching any device.
///
/// Fails only when a backend was forced without fallback and cannot be used.
pub fn resolve(config: &AccelConfig, inventory: &Inventory) -> Result<Plan, AccelerationError> {
    let mut plan = Plan::default();
    for backend in config.attempt_order() {
        match locate(backend, inventory) {
            Ok(candidate) => plan.candidates.push(candidate),
            Err(reason) => plan.skipped.push(AccelAttempt {
                backend,
                outcome: AttemptOutcome::Unavailable(reason),
            }),
        }
    }
    if let AccelMode::Force {
        backend,
        allow_fallback: false,
    } = config.mode
        && let Some(AccelAttempt {
            outcome: AttemptOutcome::Unavailable(reason),
            ..
        }) = plan.skipped.first()
    {
        return Err(AccelerationError::Unavailable {
            backend,
            reason: reason.clone(),
        });
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use AccelBackend::{Cpu, Cuda, Hip, Metal, Vulkan};
    use std::{collections::HashMap, path::PathBuf};

    fn device(id: i32, backend: AccelBackend, tpe: DeviceType) -> Device {
        Device {
            id,
            name: format!("{backend}{id}"),
            description: String::new(),
            vram_free: 0,
            vram_total: 0,
            tpe,
            backend: Some(backend),
            caps: None,
        }
    }

    fn inventory(devices: Vec<Device>, built: &[AccelBackend]) -> Inventory {
        Inventory {
            devices,
            build: AccelBackend::ALL
                .iter()
                .map(|b| {
                    let status = if built.contains(b) {
                        BuildStatus::Built
                    } else {
                        BuildStatus::NotBuilt
                    };
                    (*b, status)
                })
                .collect::<HashMap<_, _>>(),
            missing_runtime: HashMap::new(),
            runtime_dirs: vec![PathBuf::from("/runtime")],
        }
    }

    fn without_runtime(
        mut inventory: Inventory,
        backend: AccelBackend,
        library: &str,
    ) -> Inventory {
        inventory
            .missing_runtime
            .insert(backend, library.to_owned());
        inventory
    }

    fn cuda_and_vulkan_machine() -> Inventory {
        inventory(
            vec![
                device(0, Cuda, DeviceType::Gpu),
                device(1, Vulkan, DeviceType::Igpu),
                device(2, Vulkan, DeviceType::Gpu),
                device(3, Cpu, DeviceType::Cpu),
            ],
            &[Cuda, Vulkan, Cpu],
        )
    }

    fn backends(plan: &Plan) -> Vec<AccelBackend> {
        plan.candidates.iter().map(|c| c.backend).collect()
    }

    fn gpu_target(plan: &Plan, backend: AccelBackend) -> Option<&GpuTarget> {
        plan.candidates
            .iter()
            .find_map(|c| match (&c.target, c.backend == backend) {
                (Target::Gpu(target), true) => Some(target),
                (Target::Gpu(_) | Target::Cpu | Target::CoreMl, _) => None,
            })
    }

    #[test]
    fn auto_orders_available_backends_by_priority_and_records_skips() {
        let plan = resolve(&AccelConfig::default(), &cuda_and_vulkan_machine()).unwrap();
        assert_eq!(backends(&plan), [Cuda, Vulkan, Cpu]);
        let skipped: Vec<_> = plan.skipped.iter().map(|a| a.backend).collect();
        assert_eq!(
            skipped,
            [Hip, AccelBackend::CoreMl, Metal, AccelBackend::Sycl]
        );
    }

    #[test]
    fn gpu_ordinal_counts_only_gpu_devices() {
        let inventory = inventory(
            vec![
                device(0, AccelBackend::Cpu, DeviceType::Cpu),
                device(1, Cuda, DeviceType::Accel),
                device(2, Cuda, DeviceType::Gpu),
                device(3, Vulkan, DeviceType::Unknown),
                device(4, Vulkan, DeviceType::Igpu),
            ],
            &[Cuda, Vulkan, Cpu],
        );
        let plan = resolve(&AccelConfig::default(), &inventory).unwrap();
        let cuda = gpu_target(&plan, Cuda).unwrap();
        let vulkan = gpu_target(&plan, Vulkan).unwrap();
        assert_eq!((cuda.device_id, cuda.ordinal), (2, 0));
        assert_eq!((vulkan.device_id, vulkan.ordinal), (4, 1));
    }

    #[test]
    fn first_device_of_a_backend_is_chosen() {
        let plan = resolve(
            &AccelConfig::forced(Vulkan, false),
            &cuda_and_vulkan_machine(),
        )
        .unwrap();
        let vulkan = gpu_target(&plan, Vulkan).unwrap();
        assert_eq!((vulkan.device_id, vulkan.ordinal), (1, 1));
    }

    #[test]
    fn forced_unavailable_backend_without_fallback_is_an_error() {
        let result = resolve(
            &AccelConfig::forced(Metal, false),
            &cuda_and_vulkan_machine(),
        );
        assert!(matches!(
            result,
            Err(AccelerationError::Unavailable {
                backend: Metal,
                reason: UnavailableReason::NotBuilt,
            })
        ));
    }

    #[test]
    fn forced_built_backend_without_device_reports_no_device() {
        let inventory = inventory(vec![device(0, Cpu, DeviceType::Cpu)], &[Cuda, Cpu]);
        let result = resolve(&AccelConfig::forced(Cuda, false), &inventory);
        assert!(matches!(
            result,
            Err(AccelerationError::Unavailable {
                backend: Cuda,
                reason: UnavailableReason::NoDevice,
            })
        ));
    }

    #[test]
    fn backend_with_a_missing_runtime_library_is_skipped_by_auto() {
        let machine = without_runtime(cuda_and_vulkan_machine(), Cuda, "cublas64_13.dll");
        let plan = resolve(&AccelConfig::default(), &machine).unwrap();
        assert_eq!(backends(&plan), [Vulkan, Cpu]);
        assert_eq!(
            plan.skipped.first(),
            Some(&AccelAttempt {
                backend: Cuda,
                outcome: AttemptOutcome::Unavailable(UnavailableReason::RuntimeLibraryMissing {
                    library: "cublas64_13.dll".to_owned(),
                    searched: vec![PathBuf::from("/runtime")],
                }),
            })
        );
    }

    #[test]
    fn forced_backend_with_a_missing_runtime_library_is_an_error_despite_its_device() {
        let machine = without_runtime(cuda_and_vulkan_machine(), Cuda, "cublas64_13.dll");
        let result = resolve(&AccelConfig::forced(Cuda, false), &machine);
        assert!(matches!(
            result,
            Err(AccelerationError::Unavailable {
                backend: Cuda,
                reason: UnavailableReason::RuntimeLibraryMissing { .. },
            })
        ));
    }

    #[test]
    fn forced_backend_with_a_missing_runtime_library_falls_through_when_allowed() {
        let machine = without_runtime(cuda_and_vulkan_machine(), Cuda, "cublas64_13.dll");
        let plan = resolve(&AccelConfig::forced(Cuda, true), &machine).unwrap();
        assert_eq!(backends(&plan), [Vulkan, Cpu]);
    }

    #[test]
    fn forced_unavailable_backend_with_fallback_falls_through() {
        let plan = resolve(
            &AccelConfig::forced(Metal, true),
            &cuda_and_vulkan_machine(),
        )
        .unwrap();
        assert_eq!(backends(&plan), [Cuda, Vulkan, Cpu]);
        assert_eq!(plan.skipped.first().map(|a| a.backend), Some(Metal));
    }

    #[test]
    fn forced_available_backend_with_fallback_goes_first() {
        let plan = resolve(
            &AccelConfig::forced(Vulkan, true),
            &cuda_and_vulkan_machine(),
        )
        .unwrap();
        assert_eq!(backends(&plan), [Vulkan, Cuda, Cpu]);
    }

    #[test]
    fn forced_available_backend_without_fallback_is_alone() {
        let plan = resolve(
            &AccelConfig::forced(Cuda, false),
            &cuda_and_vulkan_machine(),
        )
        .unwrap();
        assert_eq!(backends(&plan), [Cuda]);
        assert_eq!(plan.skipped, []);
    }

    #[test]
    fn cpu_only_machine_resolves_to_cpu() {
        let inventory = inventory(vec![device(0, Cpu, DeviceType::Cpu)], &[Cpu]);
        let plan = resolve(&AccelConfig::default(), &inventory).unwrap();
        assert_eq!(backends(&plan), [Cpu]);
    }

    #[test]
    fn custom_priority_reorders_candidates() {
        let config = AccelConfig::default().with_priority(vec![Vulkan, Cuda]);
        let plan = resolve(&config, &cuda_and_vulkan_machine()).unwrap();
        assert_eq!(backends(&plan), [Vulkan, Cuda, Cpu]);
    }

    #[test]
    fn unknown_build_status_is_reported_as_such() {
        let inventory = Inventory {
            devices: vec![device(0, Cpu, DeviceType::Cpu)],
            build: HashMap::new(),
            missing_runtime: HashMap::new(),
            runtime_dirs: Vec::new(),
        };
        let result = resolve(&AccelConfig::forced(Hip, false), &inventory);
        assert!(matches!(
            result,
            Err(AccelerationError::Unavailable {
                backend: Hip,
                reason: UnavailableReason::NotBuiltOrNoDevice,
            })
        ));
    }
}
