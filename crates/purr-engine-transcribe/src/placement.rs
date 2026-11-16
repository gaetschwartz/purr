//! Where a model is loaded: the backend and device transcribe.cpp is asked for.

use purr_core::{
    accel::{backend::AccelBackend, plan::Candidate},
    dev::Device,
    engine::error::OpenError,
};
use transcribe_cpp::{Backend, ModelOptions};

/// The backend transcribe.cpp is asked for to run on `backend`, or `None` when it has no such
/// backend.
const fn requested(backend: AccelBackend) -> Option<Backend> {
    match backend {
        AccelBackend::Cpu => Some(Backend::Cpu),
        AccelBackend::Cuda => Some(Backend::Cuda),
        AccelBackend::Hip => Some(Backend::Rocm),
        AccelBackend::Metal => Some(Backend::Metal),
        AccelBackend::Vulkan => Some(Backend::Vulkan),
        AccelBackend::CoreMl | AccelBackend::Sycl => None,
    }
}

/// The position in transcribe.cpp's registry of the GPU `candidate` stands for among `devices`,
/// which this engine listed: a device's `id` is its position in the registry's list.
fn registry_index(candidate: &Candidate, devices: &[Device]) -> Option<usize> {
    let device = candidate.find_gpu(devices)?;
    usize::try_from(device.id).ok()
}

/// The options that load a model on `candidate`: the processor strictly, or exactly the GPU the
/// candidate names, which transcribe.cpp does not replace by another device when it fails.
pub fn model_options(candidate: &Candidate, devices: &[Device]) -> Result<ModelOptions, OpenError> {
    let backend = requested(candidate.backend).ok_or(OpenError::NoDevice)?;
    if candidate.backend == AccelBackend::Cpu {
        return Ok(ModelOptions {
            backend,
            ..ModelOptions::default()
        });
    }
    let index = registry_index(candidate, devices).ok_or(OpenError::NoDevice)?;
    let device = transcribe_cpp::devices()
        .into_iter()
        .nth(index)
        .ok_or(OpenError::NoDevice)?;
    Ok(ModelOptions {
        backend,
        device: Some(device),
        ..ModelOptions::default()
    })
}

#[cfg(test)]
mod tests {
    use purr_common::{devices::DeviceKey, platform::DeviceType};

    use super::*;

    fn gpu(id: i32, name: &str, pci: &str) -> Device {
        Device {
            id,
            name: format!("Vulkan{id}"),
            description: name.to_owned(),
            device_id: Some(pci.to_owned()),
            vram_free: 0,
            vram_total: 0,
            tpe: DeviceType::Gpu,
            backend: Some(AccelBackend::Vulkan),
            caps: None,
            vulkan_driver: None,
        }
    }

    fn candidate(backend: AccelBackend, name: &str, pci: Option<&str>) -> Candidate {
        Candidate {
            key: DeviceKey::cpu(),
            backend,
            name: name.to_owned(),
            pci_id: pci.map(str::to_owned),
        }
    }

    #[test]
    fn every_backend_transcribe_cpp_has_is_requested_by_its_own_name() {
        assert_eq!(requested(AccelBackend::Vulkan), Some(Backend::Vulkan));
        assert_eq!(requested(AccelBackend::Hip), Some(Backend::Rocm));
        assert_eq!(requested(AccelBackend::Cpu), Some(Backend::Cpu));
    }

    #[test]
    fn coreml_and_sycl_are_not_backends_of_this_engine() {
        assert_eq!(requested(AccelBackend::CoreMl), None);
        assert_eq!(requested(AccelBackend::Sycl), None);
    }

    #[test]
    fn the_processor_needs_no_device() {
        let options = model_options(&candidate(AccelBackend::Cpu, "CPU", None), &[]).unwrap();
        assert_eq!(options.backend, Backend::Cpu);
        assert!(options.device.is_none());
    }

    #[test]
    fn a_gpu_this_engine_does_not_list_is_no_device() {
        let listed = [gpu(0, "Card", "0000:05:00.0")];
        let wanted = candidate(AccelBackend::Vulkan, "Card", Some("0000:01:00.0"));
        assert!(matches!(
            model_options(&wanted, &listed),
            Err(OpenError::NoDevice)
        ));
    }

    #[test]
    fn the_registry_index_is_that_of_the_matching_listed_gpu() {
        let listed = [
            gpu(0, "Other", "0000:02:00.0"),
            gpu(1, "Card", "0000:01:00.0"),
        ];
        let wanted = candidate(AccelBackend::Vulkan, "Card", Some("0000:01:00.0"));
        assert_eq!(registry_index(&wanted, &listed), Some(1));
    }
}
