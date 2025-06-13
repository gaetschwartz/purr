//! GPU detection and information

use serde::{Deserialize, Serialize};
use whisper_rs::whisper_rs_sys::ggml_backend_dev_count;

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    pub id: i32,
    pub name: String,
    pub vram_free: usize,
    pub vram_total: usize,
    pub tpe: DeviceType, // Type of GPU (e.g., "Vulkan", "CUDA")
}

/// GPU acceleration status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStatus {
    pub vulkan_available: bool,
    pub cuda_available: bool,
    pub coreml_available: bool,
    pub devices: Vec<Device>,
}

/// List available GPU devices
pub fn list_devices() -> Vec<Device> {
    // TODO: maybe use ggml_backend_dev_count instead?
    #[allow(unused_mut)]
    let mut devices = Vec::new();
    #[cfg(feature = "vulkan")]
    {
        devices.extend(
            whisper_rs::vulkan::list_devices()
                .into_iter()
                .map(|dev| Device {
                    id: dev.id,
                    name: dev.name,
                    vram_free: dev.vram.free,
                    vram_total: dev.vram.total,
                    tpe: DeviceType::Vulkan,
                }),
        )
    }

    let cnt = unsafe { ggml_backend_dev_count() };
    for i in 0..cnt {
        let device = unsafe { whisper_rs::whisper_rs_sys::ggml_backend_dev_get(i) };
        if device.is_null() {
            continue;
        }
        let name = unsafe {
            let c_str =
                std::ffi::CStr::from_ptr(whisper_rs::whisper_rs_sys::ggml_backend_dev_name(device));
            c_str.to_string_lossy().into_owned()
        };
        let desc = unsafe {
            let c_str = std::ffi::CStr::from_ptr(
                whisper_rs::whisper_rs_sys::ggml_backend_dev_description(device),
            );
            c_str.to_string_lossy().into_owned()
        };
        let mut free = 0;
        let mut total = 0;
        unsafe {
            whisper_rs::whisper_rs_sys::ggml_backend_dev_memory(device, &mut free, &mut total)
        };
        let tpe = unsafe { whisper_rs::whisper_rs_sys::ggml_backend_dev_type(device) };
        let tpe = match tpe {
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
            name: format!("{} ({})", name, desc),
            vram_free: free,
            vram_total: total,
            tpe,
        });
    }
    devices
}

/// Check GPU acceleration status
pub fn check_gpu_status() -> GpuStatus {
    let devices = list_devices();

    GpuStatus {
        vulkan_available: cfg!(feature = "vulkan") && !devices.is_empty(),
        cuda_available: cfg!(feature = "cuda"),
        coreml_available: cfg!(feature = "coreml"),
        devices,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, strum::FromRepr)]
#[repr(u8)]
pub enum DeviceType {
    Cpu = 0,
    Gpu = 1,
    Accel = 2,
}
