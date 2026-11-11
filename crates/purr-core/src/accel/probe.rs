use whisper_rs::whisper_rs_sys::{ggml_backend_dev_get, ggml_backend_dev_init, ggml_backend_free};

use super::{plan::GpuTarget, report::AttemptFailure};

/// Initialise the device and release it again.
///
/// whisper.cpp falls back to the CPU without telling its caller when a GPU backend fails to
/// initialise, so a successful context creation does not prove the GPU is in use.
pub fn probe_gpu(target: &GpuTarget) -> Result<(), AttemptFailure> {
    let failure = || AttemptFailure::DeviceInit {
        device: target.name.clone(),
    };
    let index = usize::try_from(target.device_id).map_err(|_negative_id| failure())?;
    let device = unsafe { ggml_backend_dev_get(index) };
    if device.is_null() {
        return Err(failure());
    }
    let backend = unsafe { ggml_backend_dev_init(device, std::ptr::null()) };
    if backend.is_null() {
        return Err(failure());
    }
    unsafe { ggml_backend_free(backend) };
    Ok(())
}
