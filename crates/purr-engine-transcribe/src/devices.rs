use purr_common::platform::DeviceType;
use purr_core::{accel::backend::AccelBackend, dev::Device};

/// The devices transcribe.cpp has registered in this process, in its order; a device's `id` is its
/// index in that list.
#[must_use]
pub fn list_devices() -> Vec<Device> {
    transcribe_cpp::devices()
        .iter()
        .enumerate()
        .map(|(index, device)| Device {
            id: index as i32,
            name: device.name.clone(),
            description: device.description.clone(),
            device_id: device.device_id.clone(),
            vram_free: device.memory_free as usize,
            vram_total: device.memory_total as usize,
            tpe: device_type(device.device_type),
            backend: backend_of(&device.kind),
            caps: None,
            vulkan_driver: None,
        })
        .collect()
}

pub(crate) const fn device_type(device_type: transcribe_cpp::DeviceType) -> DeviceType {
    match device_type {
        transcribe_cpp::DeviceType::Cpu => DeviceType::Cpu,
        transcribe_cpp::DeviceType::Gpu => DeviceType::Gpu,
        transcribe_cpp::DeviceType::Igpu => DeviceType::Igpu,
        transcribe_cpp::DeviceType::Accel => DeviceType::Accel,
        transcribe_cpp::DeviceType::Unknown => DeviceType::Unknown,
    }
}

/// `kind` is the vendor classification transcribe.cpp documents for its devices.
pub(crate) fn backend_of(kind: &str) -> Option<AccelBackend> {
    match kind {
        "cuda" => Some(AccelBackend::Cuda),
        "rocm" => Some(AccelBackend::Hip),
        "metal" => Some(AccelBackend::Metal),
        "sycl" => Some(AccelBackend::Sycl),
        "vulkan" => Some(AccelBackend::Vulkan),
        "cpu" => Some(AccelBackend::Cpu),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_vendor_kinds_map_to_backends() {
        assert_eq!(backend_of("vulkan"), Some(AccelBackend::Vulkan));
        assert_eq!(backend_of("rocm"), Some(AccelBackend::Hip));
        assert_eq!(backend_of("cpu"), Some(AccelBackend::Cpu));
    }

    #[test]
    fn kinds_purr_does_not_select_between_have_no_backend() {
        for kind in ["accel", "gpu", "unknown", ""] {
            assert_eq!(backend_of(kind), None, "{kind}");
        }
    }

    #[test]
    fn the_device_types_keep_their_class() {
        assert_eq!(
            device_type(transcribe_cpp::DeviceType::Igpu),
            DeviceType::Igpu
        );
        assert_eq!(
            device_type(transcribe_cpp::DeviceType::Accel),
            DeviceType::Accel
        );
    }
}
