//! GPU detection and information

pub mod caps;

use std::{collections::HashMap, path::PathBuf, sync::Arc};

use purr_common::{accel::CpuReport, platform::DeviceType, vulkan::VulkanDriverStatus};
use serde::{Deserialize, Serialize};

use self::caps::DeviceCaps;
use crate::{
    accel::{
        backend::AccelBackend,
        devices::DetectedDevice,
        engine_report::{EngineReport, reports},
        hardware,
        report::UnavailableReason,
    },
    cpu,
    engine::Engine,
};

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    /// The device's position in the inventory it was enumerated in: the index `purr sys` prints.
    pub id: i32,
    pub name: String,
    pub description: String, // Optional description field
    /// The id the device reports; for PCI devices the bus id as "domain:bus:device.function".
    pub device_id: Option<String>,
    pub vram_free: usize,
    pub vram_total: usize,
    pub tpe: DeviceType,
    /// Backend owning the device; `None` for backends purr does not select between (BLAS).
    pub backend: Option<AccelBackend>,
    pub caps: Option<DeviceCaps>,
    /// What the Vulkan driver says about a Vulkan device; `None` for every other device.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vulkan_driver: Option<VulkanDriverStatus>,
}

impl Device {
    /// The capabilities the device reports as set; `None` when it reported none.
    #[must_use]
    pub fn capability_names(&self) -> Option<Vec<&'static str>> {
        self.caps.map(DeviceCaps::names)
    }
}

/// GPU acceleration status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Whether each backend can run on any loaded engine.
    pub backends: HashMap<AccelBackend, FeatureStatus>,
    pub cpu: CpuReport,
    /// The index of the GPU each usable GPU backend picks when the device is not chosen, in the
    /// first engine's list.
    pub picked_gpus: HashMap<AccelBackend, i32>,
    /// The indexes of the devices transcription tries when nobody arranged them, first to last,
    /// in the first engine's list.
    #[serde(default)]
    pub default_order: Vec<i32>,
    /// The first engine's own devices.
    pub devices: Vec<Device>,
}

impl SystemInfo {
    /// Check system capabilities and the devices of `engines`.
    #[must_use]
    pub fn get(engines: &[Arc<dyn Engine>]) -> Self {
        Self::of(&reports(engines, &[]))
    }

    /// What `reports` say together. The devices and their indexes are the first report's, the ones
    /// `--device` takes for that engine; the backends and the processor span every loaded engine.
    #[must_use]
    pub fn of(reports: &[EngineReport]) -> Self {
        let first = reports.first();
        let detected: &[DetectedDevice] = first.map_or(&[], |report| &report.detected);
        let inventory = first
            .map(|report| report.inventory.clone())
            .unwrap_or_default();

        Self {
            picked_gpus: AccelBackend::ALL
                .iter()
                .filter_map(|backend| {
                    detected
                        .iter()
                        .find(|device| device.backend == *backend && backend.is_gpu())
                        .and_then(|device| device.index)
                        .map(|id| (*backend, id))
                })
                .collect(),
            default_order: detected.iter().filter_map(|device| device.index).collect(),
            backends: AccelBackend::ALL
                .iter()
                .map(|backend| {
                    let status = match hardware::backend_status(*backend, reports) {
                        Ok(()) => FeatureStatus::Available(None),
                        Err(UnavailableReason::RuntimeLibraryMissing { library, searched }) => {
                            FeatureStatus::RuntimeLibraryMissing { library, searched }
                        }
                        Err(UnavailableReason::NoDevice) => FeatureStatus::EnabledButNotAvailable,
                        Err(
                            UnavailableReason::NotBuilt | UnavailableReason::NotBuiltOrNoDevice,
                        ) => FeatureStatus::Disabled,
                    };
                    (*backend, status)
                })
                .collect(),
            cpu: cpu::report(&inventory),
            devices: inventory.devices,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum FeatureStatus {
    Disabled,
    RuntimeLibraryMissing {
        library: String,
        #[serde(default)]
        searched: Vec<PathBuf>,
    },
    EnabledButNotAvailable,
    Available(Option<u16>),
}

impl SystemInfo {
    /// Returns devices capable of acceleration
    pub fn devices_can_accelerate(&self) -> impl Iterator<Item = &Device> {
        self.devices.iter().filter(|d| {
            matches!(
                d.tpe,
                DeviceType::Gpu | DeviceType::Igpu | DeviceType::Accel
            )
        })
    }

    /// Returns devices capable of GPU acceleration
    pub fn gpus(&self) -> impl Iterator<Item = &Device> {
        self.devices.iter().filter(|d| d.tpe.is_gpu())
    }

    /// Returns Accelerator devices (e.g., TPU, NPU)
    pub fn accelerators_exclude_cpu(&self) -> impl Iterator<Item = &Device> {
        self.devices.iter().filter(|d| d.tpe == DeviceType::Accel)
    }

    /// Returns CPU devices
    pub fn cpus(&self) -> impl Iterator<Item = &Device> {
        self.devices.iter().filter(|d| d.tpe == DeviceType::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(caps: Option<DeviceCaps>) -> Device {
        Device {
            id: 0,
            name: "Vulkan0".to_owned(),
            description: String::new(),
            device_id: None,
            vram_free: 0,
            vram_total: 0,
            tpe: DeviceType::Gpu,
            backend: Some(AccelBackend::Vulkan),
            caps,
            vulkan_driver: None,
        }
    }

    #[test]
    fn capabilities_are_listed_by_name() {
        let caps = DeviceCaps {
            asynchronous: true,
            host_buffer: false,
            buffer_from_host_ptr: true,
            events: true,
        };
        assert_eq!(
            device(Some(caps)).capability_names(),
            Some(vec!["async", "buffer_from_host_ptr", "events"])
        );
    }

    #[test]
    fn a_device_with_no_capabilities_set_lists_none() {
        let caps = DeviceCaps {
            asynchronous: false,
            host_buffer: false,
            buffer_from_host_ptr: false,
            events: false,
        };
        assert_eq!(device(Some(caps)).capability_names(), Some(vec![]));
    }

    #[test]
    fn a_device_without_reported_capabilities_has_no_names() {
        assert_eq!(device(None).capability_names(), None);
    }
}
