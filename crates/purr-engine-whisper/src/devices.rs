use purr_core::{
    accel::{devices::display_name, plan::Candidate},
    dev::Device,
};

/// A GPU as whisper.cpp addresses it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuHandle {
    /// Position of the device in ggml's global device list.
    pub device_id: i32,
    /// Value for whisper's `gpu_device`: whisper counts only GPU and iGPU devices, in global
    /// order, so this differs from `device_id` as soon as another kind of device comes first.
    pub ordinal: i32,
    pub name: String,
}

/// The GPU of `devices`, ggml's own list, that `candidate` stands for.
pub fn locate(candidate: &Candidate, devices: &[Device]) -> Option<GpuHandle> {
    let device = candidate.find_gpu(devices)?;
    let ordinal = devices
        .iter()
        .filter(|other| other.tpe.is_gpu())
        .position(|other| other.id == device.id)?;
    Some(GpuHandle {
        device_id: device.id,
        ordinal: i32::try_from(ordinal).ok()?,
        name: display_name(device),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use purr_common::{devices::DeviceOrder, platform::DeviceType};
    use purr_core::accel::{
        backend::{
            AccelBackend::{self, Cpu, Cuda, Vulkan},
            BuildStatus,
        },
        build_info::BuildInfo,
        inventory::Inventory,
        plan::resolve,
    };

    fn device(id: i32, name: &str, backend: AccelBackend, tpe: DeviceType) -> Device {
        Device {
            id,
            name: format!("{backend}{id}"),
            description: name.to_owned(),
            device_id: None,
            vram_free: 0,
            vram_total: 0,
            tpe,
            backend: Some(backend),
            caps: None,
            vulkan_driver: None,
        }
    }

    fn with_pci(mut device: Device, pci: &str) -> Device {
        device.device_id = Some(pci.to_owned());
        device
    }

    fn inventory(devices: Vec<Device>, built: &[AccelBackend]) -> Inventory {
        Inventory {
            devices,
            build: BuildInfo {
                backends: AccelBackend::ALL
                    .iter()
                    .map(|backend| {
                        let status = if built.contains(backend) {
                            BuildStatus::Built
                        } else {
                            BuildStatus::NotBuilt
                        };
                        (*backend, status)
                    })
                    .collect::<HashMap<_, _>>(),
                cpu: Vec::new(),
            },
            ..Inventory::default()
        }
    }

    fn candidates(devices: Vec<Device>) -> Vec<Candidate> {
        let machine = inventory(devices, &[Cuda, Vulkan, Cpu]);
        resolve(&DeviceOrder::automatic(), &machine)
            .candidates()
            .cloned()
            .collect()
    }

    fn candidate_of(candidates: &[Candidate], backend: AccelBackend, name: &str) -> Candidate {
        candidates
            .iter()
            .find(|candidate| candidate.backend == backend && candidate.name == name)
            .cloned()
            .expect("the candidate exists")
    }

    fn located(handle: Option<GpuHandle>) -> Option<(i32, i32)> {
        handle.map(|handle| (handle.device_id, handle.ordinal))
    }

    #[test]
    fn the_ordinal_counts_gpus_of_every_backend_in_ggml_order() {
        let devices = vec![
            device(0, "Processor", Cpu, DeviceType::Cpu),
            device(1, "NPU", Cuda, DeviceType::Accel),
            device(2, "RTX", Cuda, DeviceType::Gpu),
            device(3, "Odd", Vulkan, DeviceType::Unknown),
            device(4, "Radeon", Vulkan, DeviceType::Igpu),
        ];
        let found = candidates(devices.clone());
        assert_eq!(
            located(locate(&candidate_of(&found, Cuda, "RTX"), &devices)),
            Some((2, 0))
        );
        assert_eq!(
            located(locate(&candidate_of(&found, Vulkan, "Radeon"), &devices)),
            Some((4, 1))
        );
    }

    #[test]
    fn devices_with_ids_are_found_by_id() {
        let devices = vec![
            with_pci(
                device(0, "Same Card", Vulkan, DeviceType::Gpu),
                "0000:01:00.0",
            ),
            with_pci(
                device(1, "Same Card", Vulkan, DeviceType::Gpu),
                "0000:02:00.0",
            ),
        ];
        let found = candidates(devices.clone());
        let second = found
            .iter()
            .find(|candidate| candidate.pci_id.as_deref() == Some("0000:02:00.0"))
            .expect("the second card is a candidate");
        assert_eq!(located(locate(second, &devices)), Some((1, 1)));
    }

    #[test]
    fn devices_without_ids_are_found_by_name_and_position_among_that_name() {
        let devices = vec![
            device(0, "Same Card", Vulkan, DeviceType::Gpu),
            device(1, "Other Card", Vulkan, DeviceType::Gpu),
            device(2, "Same Card", Vulkan, DeviceType::Gpu),
        ];
        let found = candidates(devices.clone());
        let same: Vec<_> = found
            .iter()
            .filter(|candidate| candidate.name == "Same Card")
            .map(|candidate| located(locate(candidate, &devices)))
            .collect();
        assert_eq!(same, [Some((0, 0)), Some((2, 2))]);
        assert_eq!(
            located(locate(
                &candidate_of(&found, Vulkan, "Other Card"),
                &devices
            )),
            Some((1, 1))
        );
    }

    #[test]
    fn a_device_the_engine_does_not_list_is_not_found() {
        let found = candidates(vec![
            with_pci(device(0, "Card", Vulkan, DeviceType::Gpu), "0000:01:00.0"),
            device(1, "Bare Card", Cuda, DeviceType::Gpu),
        ]);
        let engine = vec![
            with_pci(device(0, "Card", Vulkan, DeviceType::Gpu), "0000:05:00.0"),
            device(1, "Processor", Cpu, DeviceType::Cpu),
        ];
        assert_eq!(
            located(locate(&candidate_of(&found, Vulkan, "Card"), &engine)),
            None
        );
        assert_eq!(
            located(locate(&candidate_of(&found, Cuda, "Bare Card"), &engine)),
            None
        );
    }

    #[test]
    fn an_engine_that_orders_its_devices_differently_still_finds_the_device() {
        let found = candidates(vec![
            with_pci(device(0, "RTX", Cuda, DeviceType::Gpu), "0000:01:00.0"),
            with_pci(device(1, "RTX", Vulkan, DeviceType::Gpu), "0000:01:00.0"),
            device(2, "Radeon", Vulkan, DeviceType::Igpu),
            device(3, "Processor", Cpu, DeviceType::Cpu),
        ]);
        let engine = vec![
            device(0, "Processor", Cpu, DeviceType::Cpu),
            device(1, "Radeon", Vulkan, DeviceType::Igpu),
            with_pci(device(2, "RTX", Vulkan, DeviceType::Gpu), "0000:01:00.0"),
            with_pci(device(3, "RTX", Cuda, DeviceType::Gpu), "0000:01:00.0"),
        ];
        assert_eq!(
            located(locate(&candidate_of(&found, Vulkan, "RTX"), &engine)),
            Some((2, 1))
        );
        assert_eq!(
            located(locate(&candidate_of(&found, Cuda, "RTX"), &engine)),
            Some((3, 2))
        );
        assert_eq!(
            located(locate(&candidate_of(&found, Vulkan, "Radeon"), &engine)),
            Some((1, 0))
        );
    }
}
