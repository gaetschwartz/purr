use crate::platform;
use dioxus::prelude::*;
use purr_common::platform::{DeviceType, ModelInfo, Platform as _};

/// What the GPU can do for transcription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuSupport {
    Checking,
    Unavailable,
    Available { name: String, shader_f16: bool },
}

/// Models the platform can run, and the GPU capabilities they depend on.
#[derive(Clone, Copy, PartialEq)]
pub struct ModelsHandle {
    pub models: ReadSignal<Option<Result<Vec<ModelInfo>, String>>>,
    pub gpu: ReadSignal<GpuSupport>,
    refresh: Signal<u32>,
}

impl ModelsHandle {
    /// Reloads the model list, for example after a download finished.
    pub fn refresh(mut self) {
        self.refresh += 1;
    }

    /// The model with `id`, once the list has loaded.
    #[must_use]
    pub fn find(&self, id: &str) -> Option<ModelInfo> {
        self.models
            .read()
            .as_ref()?
            .as_ref()
            .ok()?
            .iter()
            .find(|model| model.id == id)
            .cloned()
    }
}

pub fn use_models() -> ModelsHandle {
    let refresh = use_signal(|| 0u32);
    let mut models = use_signal(|| None::<Result<Vec<ModelInfo>, String>>);
    let mut gpu = use_signal(|| GpuSupport::Checking);

    use_future(move || async move {
        let platform = platform::get_platform();
        let support = match platform.list_available_devices().await {
            Ok(devices) => devices
                .into_iter()
                .find(|device| device.device_type == DeviceType::Gpu)
                .map_or(GpuSupport::Unavailable, |device| GpuSupport::Available {
                    shader_f16: device
                        .capabilities
                        .as_ref()
                        .is_some_and(|capabilities| capabilities.contains_key("shader-f16")),
                    name: device.name,
                }),
            Err(error) => {
                tracing::warn!("Could not detect the GPU: {error}");
                GpuSupport::Unavailable
            }
        };
        gpu.set(support);
    });

    use_resource(move || async move {
        refresh();
        let result = platform::get_platform()
            .list_available_models()
            .await
            .map_err(|error| error.to_string());
        models.set(Some(result));
    });

    ModelsHandle {
        models: models.into(),
        gpu: gpu.into(),
        refresh,
    }
}
