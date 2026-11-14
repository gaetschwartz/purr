use std::ffi::CStr;

use purr_common::platform::DeviceType;
use whisper_rs::whisper_rs_sys as ggml;

use purr_core::{
    accel::backend::AccelBackend,
    dev::{Device, caps::DeviceCaps},
};

/// The devices whisper.cpp's ggml has registered in this process, in ggml's order; a device's
/// `id` is its index in that list.
#[must_use]
pub fn list_devices() -> Vec<Device> {
    let mut devices = Vec::new();

    let cnt = unsafe { ggml::ggml_backend_dev_count() };
    for i in 0..cnt {
        let device = unsafe { ggml::ggml_backend_dev_get(i) };
        if device.is_null() {
            continue;
        }
        let mut props: ggml::ggml_backend_dev_props = unsafe { std::mem::zeroed() };
        unsafe {
            ggml::ggml_backend_dev_get_props(device, &raw mut props);
        }
        let tpe = match props.type_ {
            ggml::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU => DeviceType::Cpu,
            ggml::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU => DeviceType::Gpu,
            ggml::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU => DeviceType::Igpu,
            ggml::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_ACCEL => DeviceType::Accel,
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
            caps: Some(DeviceCaps {
                asynchronous: props.caps.async_,
                host_buffer: props.caps.host_buffer,
                buffer_from_host_ptr: props.caps.buffer_from_host_ptr,
                events: props.caps.events,
            }),
            backend: backend_of(device),
            vulkan_driver: None,
            tpe,
        });
    }
    devices
}

fn backend_of(device: ggml::ggml_backend_dev_t) -> Option<AccelBackend> {
    let reg = unsafe { ggml::ggml_backend_dev_backend_reg(device) };
    if reg.is_null() {
        return None;
    }
    let name = unsafe { ggml::ggml_backend_reg_name(reg) };
    if name.is_null() {
        return None;
    }
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    backend_from_registry_name(&name)
}

fn backend_from_registry_name(name: &str) -> Option<AccelBackend> {
    match name {
        "CUDA" => Some(AccelBackend::Cuda),
        "ROCm" => Some(AccelBackend::Hip),
        "Metal" => Some(AccelBackend::Metal),
        "SYCL" => Some(AccelBackend::Sycl),
        "Vulkan" => Some(AccelBackend::Vulkan),
        "CPU" => Some(AccelBackend::Cpu),
        _ => None,
    }
}
