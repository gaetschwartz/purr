//! GPU detection and information

use std::{collections::HashMap, ffi::CStr, path::PathBuf};

use purr_common::{accel::CpuReport, vulkan::VulkanDriverStatus};
use serde::{Deserialize, Serialize};
use whisper_rs::whisper_rs_sys::{ggml_backend_dev_caps, ggml_backend_dev_count};

use crate::{
    accel::{
        backend::AccelBackend,
        inventory::Inventory,
        plan::{Candidate, Target, locate},
        report::UnavailableReason,
        vulkan_probe,
    },
    cpu,
};

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    pub id: i32,
    pub name: String,
    pub description: String, // Optional description field
    /// The id ggml reports; for PCI devices the bus id as "domain:bus:device.function".
    pub device_id: Option<String>,
    pub vram_free: usize,
    pub vram_total: usize,
    pub tpe: DeviceType,
    /// Backend owning the device; `None` for ggml backends purr does not select between (BLAS).
    pub backend: Option<AccelBackend>,
    #[serde(with = "serde_ggml_backend_dev_caps")]
    pub caps: Option<ggml_backend_dev_caps>, // Optional capabilities
    /// What the Vulkan driver says about a Vulkan device; `None` for every other device.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vulkan_driver: Option<VulkanDriverStatus>,
}

impl Device {
    /// The capabilities ggml reports as set, under ggml's names; `None` when it reported none.
    #[must_use]
    pub fn capability_names(&self) -> Option<Vec<&'static str>> {
        let caps = self.caps?;
        Some(
            [
                (caps.async_, "async"),
                (caps.host_buffer, "host_buffer"),
                (caps.buffer_from_host_ptr, "buffer_from_host_ptr"),
                (caps.events, "events"),
            ]
            .into_iter()
            .filter_map(|(set, name)| set.then_some(name))
            .collect(),
        )
    }
}

/// GPU acceleration status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub backends: HashMap<AccelBackend, FeatureStatus>,
    pub cpu: CpuReport,
    /// The ggml id of the GPU each usable GPU backend picks when the device is not chosen.
    pub picked_gpus: HashMap<AccelBackend, i32>,
    pub devices: Vec<Device>,
}

impl SystemInfo {
    /// Check system capabilities and available GPU devices
    #[must_use]
    pub fn get() -> Self {
        let mut inventory = Inventory::detect();
        vulkan_probe::attach_drivers(&mut inventory.devices);

        Self {
            picked_gpus: AccelBackend::ALL
                .iter()
                .filter_map(|backend| match locate(*backend, &inventory) {
                    Ok(Candidate {
                        target: Target::Gpu(gpu),
                        ..
                    }) => Some((*backend, gpu.device_id)),
                    Ok(Candidate {
                        target: Target::Cpu | Target::CoreMl,
                        ..
                    })
                    | Err(_) => None,
                })
                .collect(),
            backends: AccelBackend::ALL
                .iter()
                .map(|backend| {
                    let status = match locate(*backend, &inventory) {
                        Ok(_) => FeatureStatus::Available(None),
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
            cpu: cpu::report(&inventory.devices),
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
        self.devices
            .iter()
            .filter(|d| matches!(d.tpe, DeviceType::Gpu | DeviceType::Igpu))
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

/// List available GPU devices
#[must_use]
pub fn list_devices() -> Vec<Device> {
    #[allow(unused_mut)]
    let mut devices = Vec::new();

    let cnt = unsafe { ggml_backend_dev_count() };
    for i in 0..cnt {
        let device = unsafe { whisper_rs::whisper_rs_sys::ggml_backend_dev_get(i) };
        if device.is_null() {
            continue;
        }
        let mut props: whisper_rs::whisper_rs_sys::ggml_backend_dev_props =
            unsafe { std::mem::zeroed() };
        unsafe {
            whisper_rs::whisper_rs_sys::ggml_backend_dev_get_props(device, &raw mut props);
        }
        let tpe = match props.type_ {
            whisper_rs::whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU => {
                DeviceType::Cpu
            }
            whisper_rs::whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU => {
                DeviceType::Gpu
            }
            whisper_rs::whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU => {
                DeviceType::Igpu
            }
            whisper_rs::whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_ACCEL => {
                DeviceType::Accel
            }
            _ => DeviceType::Unknown,
        };
        devices.push(Device {
            id: i as i32,
            name: unsafe { CStr::from_ptr(props.name).to_string_lossy().into_owned() },
            description: unsafe {
                CStr::from_ptr(props.description)
                    .to_string_lossy()
                    .into_owned()
            },
            device_id: (!props.device_id.is_null()).then(|| unsafe {
                CStr::from_ptr(props.device_id)
                    .to_string_lossy()
                    .into_owned()
            }),
            vram_free: props.memory_free,
            vram_total: props.memory_total,
            caps: Some(props.caps),
            backend: backend_of(device),
            vulkan_driver: None,
            tpe,
        });
    }
    devices
}

fn backend_of(device: whisper_rs::whisper_rs_sys::ggml_backend_dev_t) -> Option<AccelBackend> {
    let reg = unsafe { whisper_rs::whisper_rs_sys::ggml_backend_dev_backend_reg(device) };
    if reg.is_null() {
        return None;
    }
    let name = unsafe { whisper_rs::whisper_rs_sys::ggml_backend_reg_name(reg) };
    if name.is_null() {
        return None;
    }
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    AccelBackend::from_ggml_registry_name(&name)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, strum::FromRepr, PartialEq, Eq)]
#[repr(u8)]
pub enum DeviceType {
    Cpu = 0,
    Gpu = 1,
    Igpu = 2,
    Accel = 3,
    Unknown = 255,
}

impl DeviceType {
    /// What kind of device it is, in plain words.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Gpu => "discrete GPU",
            Self::Igpu => "integrated GPU",
            Self::Accel => "accelerator",
            Self::Unknown => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(caps: Option<ggml_backend_dev_caps>) -> Device {
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
    fn capabilities_are_named_as_ggml_names_them() {
        let caps = ggml_backend_dev_caps {
            async_: true,
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
        let caps = ggml_backend_dev_caps {
            async_: false,
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

mod serde_ggml_backend_dev_caps {
    use serde::{Deserializer, Serializer, ser::SerializeMap as _};
    use whisper_rs::whisper_rs_sys::ggml_backend_dev_caps;

    // pub struct ggml_backend_dev_caps {
    //     pub async_: bool,
    //     pub host_buffer: bool,
    //     pub buffer_from_host_ptr: bool,
    //     pub events: bool,
    // }
    const FIELD_ASYNC: &str = "async_";
    const FIELD_HOST_BUFFER: &str = "host_buffer";
    const FIELD_BUFFER_FROM_HOST_PTR: &str = "buffer_from_host_ptr";
    const FIELD_EVENTS: &str = "events";

    pub fn serialize<S>(
        caps: &Option<ggml_backend_dev_caps>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match caps {
            Some(c) => {
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry(FIELD_ASYNC, &c.async_)?;
                map.serialize_entry(FIELD_HOST_BUFFER, &c.host_buffer)?;
                map.serialize_entry(FIELD_BUFFER_FROM_HOST_PTR, &c.buffer_from_host_ptr)?;
                map.serialize_entry(FIELD_EVENTS, &c.events)?;
                map.end()
            }
            None => serializer.serialize_none(),
        }
    }

    struct ValueVisitor;
    impl<'de> serde::de::Visitor<'de> for ValueVisitor {
        type Value = ggml_backend_dev_caps;
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a ggml_backend_dev_caps object")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut caps = ggml_backend_dev_caps {
                async_: false,
                host_buffer: false,
                buffer_from_host_ptr: false,
                events: false,
            };
            while let Some(key) = map.next_key::<String>()? {
                match key.as_str() {
                    FIELD_ASYNC => {
                        caps.async_ = map.next_value()?;
                    }
                    FIELD_HOST_BUFFER => {
                        caps.host_buffer = map.next_value()?;
                    }
                    FIELD_BUFFER_FROM_HOST_PTR => {
                        caps.buffer_from_host_ptr = map.next_value()?;
                    }
                    FIELD_EVENTS => {
                        caps.events = map.next_value()?;
                    }
                    _ => {
                        return Err(serde::de::Error::unknown_field(
                            &key,
                            &[
                                FIELD_ASYNC,
                                FIELD_HOST_BUFFER,
                                FIELD_BUFFER_FROM_HOST_PTR,
                                FIELD_EVENTS,
                            ],
                        ));
                    }
                }
            }
            Ok(caps)
        }
    }

    struct OptionVisitor;
    impl<'de> serde::de::Visitor<'de> for OptionVisitor {
        type Value = Option<ggml_backend_dev_caps>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("an optional ggml_backend_dev_caps object")
        }
        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserializer.deserialize_map(ValueVisitor).map(Some)
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<ggml_backend_dev_caps>, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_option(OptionVisitor)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde::{Deserialize, Serialize};

        #[derive(Serialize, Deserialize, Debug, Clone)]
        struct CapsWrapper(#[serde(with = "super")] Option<ggml_backend_dev_caps>);

        #[test]
        fn test_roundtrip_some() {
            let caps = ggml_backend_dev_caps {
                async_: true,
                host_buffer: false,
                buffer_from_host_ptr: true,
                events: false,
            };
            let serialized = serde_json::to_string(&CapsWrapper(Some(caps))).unwrap();
            println!("Serialized: {serialized:?}");
            let deserialized: CapsWrapper =
                serde_json::from_str(&serialized).expect("Failed to deserialize");
            println!("Deserialized: {deserialized:?}");
            assert!(deserialized.0.is_some());
            let caps = deserialized.0.unwrap();
            assert!(caps.async_);
            assert!(!caps.host_buffer);
            assert!(caps.buffer_from_host_ptr);
            assert!(!caps.events);
        }

        #[test]
        fn test_roundtrip_none() {
            let serialized = serde_json::to_string(&CapsWrapper(None)).unwrap();
            println!("Serialized: {serialized:?}");
            let deserialized: CapsWrapper =
                serde_json::from_str(&serialized).expect("Failed to deserialize");
            println!("Deserialized: {deserialized:?}");
            assert!(deserialized.0.is_none());
        }
    }
}
