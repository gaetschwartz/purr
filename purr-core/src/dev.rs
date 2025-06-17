//! GPU detection and information

use serde::{Deserialize, Serialize};
use whisper_rs::whisper_rs_sys::{self, ggml_backend_dev_caps, ggml_backend_dev_count};

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    pub id: i32,
    pub name: String,
    pub description: String, // Optional description field
    pub vram_free: usize,
    pub vram_total: usize,
    pub tpe: DeviceType, // Type of GPU (e.g., "Vulkan", "CUDA")
    #[serde(with = "serde_ggml_backend_dev_caps")]
    pub caps: Option<ggml_backend_dev_caps>, // Optional capabilities
}

/// GPU acceleration status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub vulkan_available: FeatureStatus,
    pub cuda_available: FeatureStatus,
    pub coreml_available: FeatureStatus,
    pub metal_available: FeatureStatus,
    pub system_info: SystemCapabilities,
    pub devices: Vec<Device>,
}

impl SystemInfo {
    /// Check system capabilities and available GPU devices
    pub fn get() -> SystemInfo {
        let devices = list_devices();

        SystemInfo {
            vulkan_available: {
                #[cfg(feature = "vulkan")]
                {
                    let dev_count = unsafe { whisper_rs_sys::ggml_backend_vk_get_device_count() };
                    if dev_count > 0 {
                        FeatureStatus::Available(Some(dev_count as u8))
                    } else {
                        FeatureStatus::EnabledButNotAvailable
                    }
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    FeatureStatus::Disabled
                }
            },
            cuda_available: {
                #[cfg(feature = "cuda")]
                {
                    let dev_count = unsafe { whisper_rs_sys::ggml_backend_cuda_get_device_count() };
                    if whisper_rs_sys::ggml_backend_cuda_is_available() != 0 && dev_count > 0 {
                        FeatureStatus::Available(Some(dev_count as u8))
                    } else {
                        FeatureStatus::EnabledButNotAvailable
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    FeatureStatus::Disabled
                }
            },
            coreml_available: if cfg!(feature = "coreml") {
                FeatureStatus::Available(None)
            } else {
                FeatureStatus::Disabled
            },
            metal_available: if cfg!(feature = "metal") {
                FeatureStatus::Available(None)
            } else {
                FeatureStatus::Disabled
            },
            system_info: SystemCapabilities::get(),
            devices,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum FeatureStatus {
    Disabled,
    EnabledButNotAvailable,
    Available(Option<u8>),
}

impl SystemInfo {
    /// Returns devices capable of acceleration
    pub fn devices_can_accelerate(&self) -> impl Iterator<Item = &Device> {
        self.devices
            .iter()
            .filter(|d| d.tpe == DeviceType::Gpu || d.tpe == DeviceType::Accel)
    }

    /// Returns devices capable of GPU acceleration
    pub fn gpus(&self) -> impl Iterator<Item = &Device> {
        self.devices.iter().filter(|d| d.tpe == DeviceType::Gpu)
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemCapabilities {
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
}

impl SystemCapabilities {
    pub fn get() -> Self {
        unsafe {
            Self {
                avx: whisper_rs_sys::ggml_cpu_has_avx() != 0,
                avx2: whisper_rs_sys::ggml_cpu_has_avx2() != 0,
                fma: whisper_rs_sys::ggml_cpu_has_fma() != 0,
                f16c: whisper_rs_sys::ggml_cpu_has_f16c() != 0,
            }
        }
    }
}

/// List available GPU devices
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
            whisper_rs::whisper_rs_sys::ggml_backend_dev_get_props(device, &mut props);
        }
        let tpe = match props.type_ {
            whisper_rs::whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU => {
                DeviceType::Cpu
            }
            whisper_rs::whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU => {
                DeviceType::Gpu
            }
            whisper_rs::whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_ACCEL => {
                DeviceType::Accel
            }
            _ => continue, // Skip unsupported types
        };
        devices.push(Device {
            id: i as i32,
            name: unsafe {
                std::ffi::CStr::from_ptr(props.name)
                    .to_string_lossy()
                    .into_owned()
            },
            description: unsafe {
                std::ffi::CStr::from_ptr(props.description)
                    .to_string_lossy()
                    .into_owned()
            },
            vram_free: props.memory_free,
            vram_total: props.memory_total,
            caps: Some(props.caps),
            tpe,
        });
    }
    devices
}

#[derive(Debug, Clone, Serialize, Deserialize, strum::FromRepr, PartialEq, Eq)]
#[repr(u8)]
pub enum DeviceType {
    Cpu = 0,
    Gpu = 1,
    Accel = 2,
}

mod serde_ggml_backend_dev_caps {
    use serde::{ser::SerializeMap as _, Deserializer, Serializer};
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
                        ))
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
            println!("Serialized: {:?}", serialized);
            let deserialized: CapsWrapper =
                serde_json::from_str(&serialized).expect("Failed to deserialize");
            println!("Deserialized: {:?}", deserialized);
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
            println!("Serialized: {:?}", serialized);
            let deserialized: CapsWrapper =
                serde_json::from_str(&serialized).expect("Failed to deserialize");
            println!("Deserialized: {:?}", deserialized);
            assert!(deserialized.0.is_none());
        }
    }
}
