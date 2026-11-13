use purr_common::devices::{DeviceKey, DeviceOrder};

use super::{
    backend::{AccelBackend, BuildStatus},
    devices::{DetectedDevice, detect, keys},
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
    pub key: DeviceKey,
    pub backend: AccelBackend,
    /// The name the system reports for the device.
    pub name: String,
    pub target: Target,
}

impl Candidate {
    fn of(device: &DetectedDevice) -> Self {
        Self {
            key: device.key.clone(),
            backend: device.backend,
            name: device.name.clone(),
            target: device.target.clone(),
        }
    }

    fn cpu() -> Self {
        Self {
            key: DeviceKey::cpu(),
            backend: AccelBackend::Cpu,
            name: "CPU".to_owned(),
            target: Target::Cpu,
        }
    }
}

/// What to do with one device of the list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    Try(Candidate),
    /// It cannot be used at all; nothing is loaded on it.
    Skip(AccelAttempt),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Plan {
    /// The devices to go through, first to last. The last step is always the processor.
    pub steps: Vec<Step>,
}

impl Plan {
    /// The devices that can be tried, in order.
    pub fn candidates(&self) -> impl Iterator<Item = &Candidate> {
        self.steps.iter().filter_map(|step| match step {
            Step::Try(candidate) => Some(candidate),
            Step::Skip(_) => None,
        })
    }
}

/// Why a backend that is built in cannot run: its runtime library cannot be loaded.
fn missing_runtime(backend: AccelBackend, inventory: &Inventory) -> Option<UnavailableReason> {
    inventory.missing_runtime.get(&backend).map(|library| {
        UnavailableReason::RuntimeLibraryMissing {
            library: library.clone(),
            searched: inventory.runtime_dirs.clone(),
        }
    })
}

/// Why `backend` has no device to offer.
fn absent_reason(backend: AccelBackend, inventory: &Inventory) -> UnavailableReason {
    missing_runtime(backend, inventory).unwrap_or_else(|| match inventory.build_status(backend) {
        BuildStatus::Built => UnavailableReason::NoDevice,
        BuildStatus::NotBuilt => UnavailableReason::NotBuilt,
        BuildStatus::Unknown => UnavailableReason::NotBuiltOrNoDevice,
    })
}

/// Whether `backend` can run on this machine, or why it cannot.
pub fn backend_status(
    backend: AccelBackend,
    inventory: &Inventory,
    detected: &[DetectedDevice],
) -> Result<(), UnavailableReason> {
    if let Some(reason) = missing_runtime(backend, inventory) {
        return Err(reason);
    }
    match backend {
        AccelBackend::Cpu => Ok(()),
        AccelBackend::CoreMl
        | AccelBackend::Cuda
        | AccelBackend::Hip
        | AccelBackend::Metal
        | AccelBackend::Sycl
        | AccelBackend::Vulkan => {
            if detected.iter().any(|device| device.backend == backend) {
                Ok(())
            } else {
                Err(absent_reason(backend, inventory))
            }
        }
    }
}

/// Decide which devices to try, in which order, without touching any of them.
///
/// The devices come from `order` merged with what the inventory holds. Switched-off devices are
/// left out. A device that cannot be used is a skipped step, with the reason. The processor
/// ends the plan: devices listed after it are never reached, and it is added when the order
/// lacks it.
#[must_use]
pub fn resolve(order: &DeviceOrder, inventory: &Inventory) -> Plan {
    let detected = detect(inventory);
    let list = order.effective(&keys(&detected));
    let mut steps = Vec::new();
    let mut ends_with_cpu = false;
    for preference in list.iter().filter(|preference| preference.enabled) {
        let backend = AccelBackend::from(preference.key.backend);
        let found = detected.iter().find(|device| device.key == preference.key);
        steps.push(match found {
            Some(device) => match missing_runtime(backend, inventory) {
                Some(reason) => skipped(preference.key.clone(), reason),
                None => Step::Try(Candidate::of(device)),
            },
            None if preference.key.is_cpu() => Step::Try(Candidate::cpu()),
            None => skipped(preference.key.clone(), absent_reason(backend, inventory)),
        });
        if preference.key.is_cpu() {
            ends_with_cpu = true;
            break;
        }
    }
    if !ends_with_cpu {
        steps.push(Step::Try(Candidate::cpu()));
    }
    Plan { steps }
}

fn skipped(key: DeviceKey, reason: UnavailableReason) -> Step {
    Step::Skip(AccelAttempt::new(key, AttemptOutcome::Unavailable(reason)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accel::devices::fixtures::{device, inventory, workstation};
    use AccelBackend::{Cpu, Cuda, Metal, Vulkan};
    use purr_common::platform::DeviceType;
    use purr_common::{
        accel::AccelBackend as Common,
        devices::{DeviceDefaults, DevicePreference},
    };
    use std::path::PathBuf;

    const RTX: &str = "NVIDIA GeForce RTX 2070 SUPER";
    const RADEON: &str = "AMD Radeon(TM) Graphics";

    fn cuda_rtx() -> DeviceKey {
        DeviceKey::new(Common::Cuda, RTX, 0)
    }

    fn vulkan_rtx() -> DeviceKey {
        DeviceKey::new(Common::Vulkan, RTX, 0)
    }

    fn radeon() -> DeviceKey {
        DeviceKey::new(Common::Vulkan, RADEON, 0)
    }

    fn on(key: DeviceKey) -> DevicePreference {
        DevicePreference::enabled(key)
    }

    fn off(key: DeviceKey) -> DevicePreference {
        DevicePreference {
            key,
            enabled: false,
        }
    }

    fn saved(list: Vec<DevicePreference>) -> DeviceOrder {
        DeviceOrder::saved(list)
    }

    fn tried(plan: &Plan) -> Vec<DeviceKey> {
        plan.candidates().map(|c| c.key.clone()).collect()
    }

    fn skipped_keys(plan: &Plan) -> Vec<DeviceKey> {
        plan.steps
            .iter()
            .filter_map(|step| match step {
                Step::Skip(attempt) => Some(attempt.key.clone()),
                Step::Try(_) => None,
            })
            .collect()
    }

    #[test]
    fn automatic_tries_every_device_in_the_usual_order_ending_with_the_cpu() {
        let plan = resolve(&DeviceOrder::automatic(), &workstation());
        assert_eq!(
            tried(&plan),
            [cuda_rtx(), vulkan_rtx(), radeon(), DeviceKey::cpu()]
        );
        assert_eq!(skipped_keys(&plan), []);
    }

    #[test]
    fn the_saved_order_decides() {
        let order = saved(vec![
            on(radeon()),
            on(cuda_rtx()),
            on(vulkan_rtx()),
            on(DeviceKey::cpu()),
        ]);
        assert_eq!(
            tried(&resolve(&order, &workstation())),
            [radeon(), cuda_rtx(), vulkan_rtx(), DeviceKey::cpu()]
        );
    }

    #[test]
    fn a_disabled_device_is_neither_tried_nor_reported() {
        let order = saved(vec![
            off(cuda_rtx()),
            on(vulkan_rtx()),
            off(radeon()),
            on(DeviceKey::cpu()),
        ]);
        let plan = resolve(&order, &workstation());
        assert_eq!(tried(&plan), [vulkan_rtx(), DeviceKey::cpu()]);
        assert_eq!(skipped_keys(&plan), []);
    }

    #[test]
    fn devices_after_the_cpu_are_not_reached_and_not_reported() {
        let order = saved(vec![
            on(DeviceKey::cpu()),
            on(DeviceKey::new(Common::Metal, "M3", 0)),
            on(cuda_rtx()),
        ]);
        let plan = resolve(&order, &workstation());
        assert_eq!(tried(&plan), [DeviceKey::cpu()]);
        assert_eq!(skipped_keys(&plan), []);
    }

    #[test]
    fn a_cpu_less_inventory_still_ends_with_a_cpu() {
        let machine = inventory(vec![device(0, RADEON, Vulkan, DeviceType::Igpu)], &[Vulkan]);
        let plan = resolve(&DeviceOrder::automatic(), &machine);
        assert_eq!(tried(&plan), [radeon(), DeviceKey::cpu()]);
    }

    #[test]
    fn a_saved_device_that_is_gone_is_skipped_with_the_reason_of_its_backend() {
        let gone_vulkan = DeviceKey::new(Common::Vulkan, "Unplugged eGPU", 0);
        let gone_metal = DeviceKey::new(Common::Metal, "M3", 0);
        let order = saved(vec![
            on(gone_vulkan.clone()),
            on(gone_metal.clone()),
            on(cuda_rtx()),
            on(DeviceKey::cpu()),
        ]);
        let plan = resolve(&order, &workstation());
        assert_eq!(
            plan.steps.first(),
            Some(&Step::Skip(AccelAttempt::new(
                gone_vulkan,
                AttemptOutcome::Unavailable(UnavailableReason::NoDevice)
            )))
        );
        assert_eq!(
            plan.steps.get(1),
            Some(&Step::Skip(AccelAttempt::new(
                gone_metal,
                AttemptOutcome::Unavailable(UnavailableReason::NotBuilt)
            )))
        );
        assert_eq!(
            tried(&plan),
            [cuda_rtx(), vulkan_rtx(), radeon(), DeviceKey::cpu()]
        );
    }

    #[test]
    fn a_device_of_a_backend_without_its_runtime_library_is_skipped() {
        let mut machine = workstation();
        machine
            .missing_runtime
            .insert(Cuda, "cublas64_13.dll".to_owned());
        let plan = resolve(&DeviceOrder::automatic(), &machine);
        assert_eq!(
            plan.steps.first(),
            Some(&Step::Skip(AccelAttempt::new(
                cuda_rtx(),
                AttemptOutcome::Unavailable(UnavailableReason::RuntimeLibraryMissing {
                    library: "cublas64_13.dll".to_owned(),
                    searched: vec![PathBuf::from("/runtime")],
                })
            )))
        );
        assert_eq!(tried(&plan), [vulkan_rtx(), radeon(), DeviceKey::cpu()]);
    }

    #[test]
    fn a_backend_of_unknown_build_status_is_reported_as_such() {
        let order = saved(vec![
            on(DeviceKey::new(Common::Hip, "Instinct", 0)),
            on(DeviceKey::cpu()),
        ]);
        let machine = Inventory {
            build: Default::default(),
            ..workstation()
        };
        let plan = resolve(&order, &machine);
        assert_eq!(
            plan.steps.first(),
            Some(&Step::Skip(AccelAttempt::new(
                DeviceKey::new(Common::Hip, "Instinct", 0),
                AttemptOutcome::Unavailable(UnavailableReason::NotBuiltOrNoDevice)
            )))
        );
    }

    #[test]
    fn identical_names_under_one_backend_are_two_devices_with_their_own_ordinals() {
        let machine = inventory(
            vec![
                device(0, "Same Card", Vulkan, DeviceType::Gpu),
                device(1, "Same Card", Vulkan, DeviceType::Gpu),
                device(2, "Processor", Cpu, DeviceType::Cpu),
            ],
            &[Vulkan, Cpu],
        );
        let second = DeviceKey::new(Common::Vulkan, "Same Card", 1);
        let first = DeviceKey::new(Common::Vulkan, "Same Card", 0);
        let plan = resolve(
            &saved(vec![
                on(second.clone()),
                on(first.clone()),
                on(DeviceKey::cpu()),
            ]),
            &machine,
        );
        let targets: Vec<_> = plan
            .candidates()
            .map(|c| match &c.target {
                Target::Gpu(gpu) => Some((gpu.device_id, gpu.ordinal)),
                Target::Cpu | Target::CoreMl => None,
            })
            .collect();
        assert_eq!(tried(&plan), [second, first, DeviceKey::cpu()]);
        assert_eq!(targets, [Some((1, 1)), Some((0, 0)), None]);
    }

    #[test]
    fn a_device_picked_for_one_run_goes_first_even_when_switched_off() {
        let order = saved(vec![on(cuda_rtx()), off(radeon()), on(DeviceKey::cpu())])
            .with_first(Some(radeon()));
        assert_eq!(
            tried(&resolve(&order, &workstation())),
            [radeon(), cuda_rtx(), vulkan_rtx(), DeviceKey::cpu()]
        );
    }

    #[test]
    fn a_new_device_is_tried_after_the_saved_gpus() {
        let order = saved(vec![on(radeon()), on(DeviceKey::cpu())]);
        assert_eq!(
            tried(&resolve(&order, &workstation())),
            [radeon(), cuda_rtx(), vulkan_rtx(), DeviceKey::cpu()]
        );
    }

    #[test]
    fn migrated_defaults_order_the_devices_nothing_has_arranged() {
        let order = DeviceOrder::automatic().with_defaults(DeviceDefaults {
            gpus_enabled: true,
            backends_first: vec![Common::Vulkan],
        });
        assert_eq!(
            tried(&resolve(&order, &workstation())),
            [vulkan_rtx(), radeon(), cuda_rtx(), DeviceKey::cpu()]
        );
        let order = DeviceOrder::automatic().with_defaults(DeviceDefaults {
            gpus_enabled: false,
            backends_first: Vec::new(),
        });
        assert_eq!(tried(&resolve(&order, &workstation())), [DeviceKey::cpu()]);
    }

    #[test]
    fn the_cpu_only_order_tries_only_the_cpu() {
        let plan = resolve(&DeviceOrder::cpu_only(), &workstation());
        assert_eq!(tried(&plan), [DeviceKey::cpu()]);
    }

    #[test]
    fn the_cpu_candidate_carries_the_name_the_system_reports() {
        let plan = resolve(&DeviceOrder::cpu_only(), &workstation());
        assert_eq!(
            plan.candidates()
                .map(|c| c.name.clone())
                .collect::<Vec<_>>(),
            ["AMD Ryzen 9 7950X 16-Core Processor"]
        );
    }

    #[test]
    fn a_backend_with_a_device_is_available_and_one_without_says_why() {
        let machine = workstation();
        let detected = detect(&machine);
        assert_eq!(backend_status(Cuda, &machine, &detected), Ok(()));
        assert_eq!(backend_status(Cpu, &machine, &detected), Ok(()));
        assert_eq!(
            backend_status(Metal, &machine, &detected),
            Err(UnavailableReason::NotBuilt)
        );
        let machine = inventory(
            vec![device(0, "Processor", Cpu, DeviceType::Cpu)],
            &[Cuda, Cpu],
        );
        let detected = detect(&machine);
        assert_eq!(
            backend_status(Cuda, &machine, &detected),
            Err(UnavailableReason::NoDevice)
        );
    }
}
