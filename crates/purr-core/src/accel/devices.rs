use std::cmp::Ordering;

use purr_common::devices::{DeviceKey, keys_in_order};

use crate::dev::Device;
use purr_common::platform::DeviceType;

use super::{
    backend::{AccelBackend, BuildStatus},
    inventory::Inventory,
    plan::{GpuTarget, Target},
};

/// A device this process can run on, with the key the user's list knows it by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectedDevice {
    pub key: DeviceKey,
    pub backend: AccelBackend,
    /// The name the system reports: ggml's description, else its device name.
    pub name: String,
    pub target: Target,
    /// The index `purr sys` prints; `None` for a device ggml does not list.
    pub ggml_id: Option<i32>,
    /// The PCI-style id ggml reports, when it does.
    pub pci_id: Option<String>,
    pub kind: Option<DeviceType>,
}

/// The name a device goes by: ggml's description, else its own name.
#[must_use]
pub fn display_name(device: &Device) -> String {
    let description = device.description.trim();
    if description.is_empty() {
        device.name.clone()
    } else {
        description.to_owned()
    }
}

fn is_gpu(device: &Device) -> bool {
    device.tpe.is_gpu()
}

/// Discrete GPUs first, then integrated ones, then the one with more memory, then the lower index.
pub fn gpu_preference(a: &Device, b: &Device) -> Ordering {
    const fn kind(device: &Device) -> u8 {
        match device.tpe {
            DeviceType::Gpu => 0,
            DeviceType::Igpu => 1,
            DeviceType::Cpu | DeviceType::Accel | DeviceType::Unknown => 2,
        }
    }
    kind(a)
        .cmp(&kind(b))
        .then_with(|| b.vram_total.cmp(&a.vram_total))
        .then_with(|| a.id.cmp(&b.id))
}

fn priority(backend: AccelBackend) -> usize {
    AccelBackend::DEFAULT_PRIORITY
        .iter()
        .position(|b| *b == backend)
        .unwrap_or(AccelBackend::DEFAULT_PRIORITY.len())
}

/// Every device the inventory holds that a transcription can run on, in the order they are
/// tried when nobody arranged them: by backend priority, then the GPU preference.
#[must_use]
pub fn detect(inventory: &Inventory) -> Vec<DetectedDevice> {
    let mut usable: Vec<(&Device, AccelBackend)> = inventory
        .devices
        .iter()
        .filter_map(|device| {
            let backend = device.backend?;
            let usable = match backend {
                AccelBackend::Cpu => device.tpe == DeviceType::Cpu,
                AccelBackend::Cuda
                | AccelBackend::Hip
                | AccelBackend::Metal
                | AccelBackend::Sycl
                | AccelBackend::Vulkan => is_gpu(device),
                AccelBackend::CoreMl => false,
            };
            usable.then_some((device, backend))
        })
        .collect();
    usable.sort_by(|(a, backend_a), (b, backend_b)| {
        priority(*backend_a)
            .cmp(&priority(*backend_b))
            .then_with(|| gpu_preference(a, b))
    });

    let mut detected: Vec<DetectedDevice> = usable
        .iter()
        .map(|(device, backend)| {
            let ordinal = inventory
                .devices
                .iter()
                .filter(|d| is_gpu(d))
                .position(|d| d.id == device.id)
                .and_then(|ordinal| i32::try_from(ordinal).ok());
            let name = display_name(device);
            DetectedDevice {
                key: DeviceKey::cpu(),
                backend: *backend,
                target: match (backend, ordinal) {
                    (AccelBackend::Cpu, _) | (_, None) => Target::Cpu,
                    (_, Some(ordinal)) => Target::Gpu(GpuTarget {
                        device_id: device.id,
                        ordinal,
                        name: name.clone(),
                    }),
                },
                name,
                ggml_id: Some(device.id),
                pci_id: device.device_id.clone(),
                kind: Some(device.tpe),
            }
        })
        .filter(|device| {
            device.backend == AccelBackend::Cpu || matches!(device.target, Target::Gpu(_))
        })
        .collect();

    if inventory.build_status(AccelBackend::CoreMl) == BuildStatus::Built {
        let at = detected
            .iter()
            .position(|d| priority(d.backend) > priority(AccelBackend::CoreMl))
            .unwrap_or(detected.len());
        detected.insert(
            at,
            DetectedDevice {
                key: DeviceKey::cpu(),
                backend: AccelBackend::CoreMl,
                name: "CoreML".to_owned(),
                target: Target::CoreMl,
                ggml_id: None,
                pci_id: None,
                kind: None,
            },
        );
    }

    let keys = keys_in_order(
        detected
            .iter()
            .map(|device| (device.backend.into(), device.name.clone())),
    );
    for (device, key) in detected.iter_mut().zip(keys) {
        device.key = key;
    }
    detected
}

/// The keys of `detected`, in order.
#[must_use]
pub fn keys(detected: &[DetectedDevice]) -> Vec<DeviceKey> {
    detected.iter().map(|device| device.key.clone()).collect()
}

#[cfg(test)]
pub(crate) mod fixtures {
    use std::{collections::HashMap, path::PathBuf};

    use super::*;

    pub fn device(id: i32, name: &str, backend: AccelBackend, tpe: DeviceType) -> Device {
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

    pub fn with_pci(mut device: Device, pci: &str) -> Device {
        device.device_id = Some(pci.to_owned());
        device
    }

    pub fn with_memory(mut device: Device, vram_total: usize) -> Device {
        device.vram_total = vram_total;
        device
    }

    pub fn inventory(devices: Vec<Device>, built: &[AccelBackend]) -> Inventory {
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

    /// An RTX card seen by CUDA and by Vulkan, an integrated Radeon and the processor.
    pub fn workstation() -> Inventory {
        inventory(
            vec![
                with_pci(
                    device(
                        0,
                        "NVIDIA GeForce RTX 2070 SUPER",
                        AccelBackend::Cuda,
                        DeviceType::Gpu,
                    ),
                    "0000:01:00.0",
                ),
                with_pci(
                    device(
                        1,
                        "NVIDIA GeForce RTX 2070 SUPER",
                        AccelBackend::Vulkan,
                        DeviceType::Gpu,
                    ),
                    "0000:01:00.0",
                ),
                device(
                    2,
                    "AMD Radeon(TM) Graphics",
                    AccelBackend::Vulkan,
                    DeviceType::Igpu,
                ),
                device(
                    3,
                    "AMD Ryzen 9 7950X 16-Core Processor",
                    AccelBackend::Cpu,
                    DeviceType::Cpu,
                ),
            ],
            &[AccelBackend::Cuda, AccelBackend::Vulkan, AccelBackend::Cpu],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;
    use AccelBackend::{Cpu, Cuda, Vulkan};
    use purr_common::accel::AccelBackend as Common;

    #[test]
    fn devices_are_listed_by_backend_priority_then_by_the_gpu_preference() {
        let inventory = inventory(
            vec![
                device(0, "Radeon iGPU", Vulkan, DeviceType::Igpu),
                device(1, "Processor", Cpu, DeviceType::Cpu),
                with_memory(device(2, "Small dGPU", Vulkan, DeviceType::Gpu), 4 << 30),
                with_memory(device(3, "Big dGPU", Vulkan, DeviceType::Gpu), 16 << 30),
                device(4, "RTX", Cuda, DeviceType::Gpu),
            ],
            &[Cuda, Vulkan, Cpu],
        );
        let names: Vec<_> = detect(&inventory)
            .into_iter()
            .map(|device| device.name)
            .collect();
        assert_eq!(
            names,
            ["RTX", "Big dGPU", "Small dGPU", "Radeon iGPU", "Processor"]
        );
    }

    #[test]
    fn keys_use_the_marketing_name_and_tell_identical_names_apart() {
        let inventory = inventory(
            vec![
                device(0, "Same Card", Vulkan, DeviceType::Gpu),
                device(1, "Same Card", Vulkan, DeviceType::Gpu),
                device(2, "Same Card", Cuda, DeviceType::Gpu),
                device(3, "Processor", Cpu, DeviceType::Cpu),
            ],
            &[Cuda, Vulkan, Cpu],
        );
        assert_eq!(
            keys(&detect(&inventory)),
            [
                DeviceKey::new(Common::Cuda, "Same Card", 0),
                DeviceKey::new(Common::Vulkan, "Same Card", 0),
                DeviceKey::new(Common::Vulkan, "Same Card", 1),
                DeviceKey::cpu(),
            ]
        );
    }

    #[test]
    fn a_name_falls_back_to_the_ggml_name_when_there_is_no_description() {
        let mut gpu = device(0, "", Vulkan, DeviceType::Gpu);
        gpu.description = "  ".to_owned();
        assert_eq!(display_name(&gpu), "vulkan0");
    }

    #[test]
    fn the_same_gpu_under_cuda_and_vulkan_is_two_devices() {
        let detected = detect(&workstation());
        let rtx: Vec<_> = detected
            .iter()
            .filter(|d| d.name == "NVIDIA GeForce RTX 2070 SUPER")
            .map(|d| d.backend)
            .collect();
        assert_eq!(rtx, [Cuda, Vulkan]);
    }

    #[test]
    fn accelerators_and_unselectable_backends_are_not_listed() {
        let mut blas = device(1, "BLAS", Cpu, DeviceType::Accel);
        blas.backend = None;
        let inventory = inventory(
            vec![
                device(0, "Processor", Cpu, DeviceType::Cpu),
                blas,
                device(2, "NPU", Cuda, DeviceType::Accel),
            ],
            &[Cuda, Cpu],
        );
        assert_eq!(keys(&detect(&inventory)), [DeviceKey::cpu()]);
    }

    #[test]
    fn a_built_coreml_is_a_device_of_its_own() {
        let inventory = inventory(
            vec![device(0, "Processor", Cpu, DeviceType::Cpu)],
            &[AccelBackend::CoreMl, Cpu],
        );
        let detected = detect(&inventory);
        assert_eq!(
            detected
                .iter()
                .map(|d| (d.backend, d.ggml_id))
                .collect::<Vec<_>>(),
            [(AccelBackend::CoreMl, None), (Cpu, Some(0))]
        );
    }

    #[test]
    fn the_gpu_ordinal_counts_gpus_of_every_backend_in_ggml_order() {
        let detected = detect(&inventory(
            vec![
                device(0, "Processor", Cpu, DeviceType::Cpu),
                device(1, "NPU", Cuda, DeviceType::Accel),
                device(2, "RTX", Cuda, DeviceType::Gpu),
                device(3, "Odd", Vulkan, DeviceType::Unknown),
                device(4, "Radeon", Vulkan, DeviceType::Igpu),
            ],
            &[Cuda, Vulkan, Cpu],
        ));
        let ordinals: Vec<_> = detected
            .iter()
            .filter_map(|d| match &d.target {
                Target::Gpu(gpu) => Some((gpu.device_id, gpu.ordinal)),
                Target::Cpu | Target::CoreMl => None,
            })
            .collect();
        assert_eq!(ordinals, [(2, 0), (4, 1)]);
    }
}
