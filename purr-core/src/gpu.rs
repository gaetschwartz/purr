//! GPU detection and information

use serde::{Deserialize, Serialize};

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub id: i32,
    pub name: String,
    pub vram_free: usize,
    pub vram_total: usize,
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
    #[cfg(feature = "vulkan")]
    {
        whisper_rs::vulkan::list_devices()
            .into_iter()
            .map(|dev| GpuDevice {
                id: dev.id,
                name: dev.name,
                vram_free: dev.vram.free,
                vram_total: dev.vram.total,
            })
            .collect()
    }
    #[cfg(feature = "cuda")]
    {
        whisper_rs::cuda::list_devices()
            .into_iter()
            .map(|dev| GpuDevice {
                id: dev.id,
                name: dev.name,
                vram_free: dev.vram.free,
                vram_total: dev.vram.total,
            })
            .collect()
    }
    #[cfg(not(feature = "vulkan"))]
    {
        Vec::new()
    }
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
