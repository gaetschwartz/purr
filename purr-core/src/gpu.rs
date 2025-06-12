//! GPU detection and information

use serde::{Deserialize, Serialize};

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
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
    pub devices: Vec<GpuDevice>,
}

/// List available GPU devices
pub fn list_gpu_devices() -> Vec<GpuDevice> {
    // TODO: maybe use ggml_backend_dev_count instead?
    #[allow(unused_mut)]
    let mut devices = Vec::new();
    #[cfg(feature = "vulkan")]
    {
        devices.extend(
            whisper_rs::vulkan::list_devices()
                .into_iter()
                .map(|dev| GpuDevice {
                    id: dev.id,
                    name: dev.name,
                    vram_free: dev.vram.free,
                    vram_total: dev.vram.total,
                    tpe: DeviceType::Vulkan,
                }),
        )
    }
    #[cfg(feature = "cuda")]
    {
        // TODO: Implement CUDA device listing
    }
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    {
        // No GPU features enabled, return empty list
    }
    devices
}

/// Check GPU acceleration status
pub fn check_gpu_status() -> GpuStatus {
    let devices = list_gpu_devices();

    GpuStatus {
        vulkan_available: cfg!(feature = "vulkan") && !devices.is_empty(),
        cuda_available: cfg!(feature = "cuda"),
        devices,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    Vulkan,
    CUDA,
}
