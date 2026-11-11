use crate::{
    components::toast::{Toasts, use_toasts},
    hooks::model_operations,
    model_catalog::{
        activity::{DownloadFailure, ModelActivity},
        entry::ModelEntry,
        listing::ModelRow,
        state::{ModelState, model_state},
    },
    platform,
    utils::delay::sleep,
};
use dioxus::{
    core::{Runtime, ScopeId, Task, spawn_forever},
    prelude::*,
};
use purr_common::platform::{DeviceType, ModelInfo, Platform as _};
use std::{collections::HashMap, time::Duration};

/// What the GPU can do for transcription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuSupport {
    Checking,
    Unavailable,
    Available { name: String, shader_f16: bool },
}

impl GpuSupport {
    /// Whether device detection found a graphics card.
    #[must_use]
    pub const fn found(&self) -> bool {
        matches!(self, Self::Available { .. })
    }
}

const ONLINE_WATCHER: &str = "const send = () => dioxus.send(navigator.onLine); send(); window.addEventListener('online', send); window.addEventListener('offline', send); await new Promise(() => {});";

const CANCEL_SETTLE: Duration = Duration::from_millis(300);

/// The models the platform can run, what is happening to them, and the actions on them.
///
/// Provided once at the app root, so downloads and checks keep running while views come and go;
/// every method may be called from any scope, including overlays and toast actions.
/// Only one download runs at a time: [`ModelsHandle::download`] does nothing while another is
/// active, so disable the button on [`ModelsHandle::is_downloading`].
#[derive(Clone, Copy, PartialEq)]
pub struct ModelsHandle {
    /// The platform's model list; `None` until the first load finishes.
    pub models: ReadSignal<Option<Result<Vec<ModelInfo>, String>>>,
    pub gpu: ReadSignal<GpuSupport>,
    /// Whether the browser or webview reports a network connection.
    pub online: ReadSignal<bool>,
    pub(super) activity: Signal<HashMap<String, ModelActivity>>,
    pub(super) tasks: Signal<HashMap<String, Task>>,
    pub(super) toasts: Toasts,
    refresh: Signal<u32>,
    gpu_check: Signal<u32>,
}

impl ModelsHandle {
    /// Looks for the graphics card again; [`Self::gpu`] changes when the answer differs.
    pub fn recheck_gpu(mut self) {
        self.gpu_check += 1;
    }

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

    /// The catalog with a state per model (subscribes the caller); `None` until the list has loaded.
    #[must_use]
    pub fn rows(&self) -> Option<Result<Vec<ModelRow>, String>> {
        let gpu = self.gpu.read();
        let activity = self.activity.read();
        let loaded = self.models.read();
        Some(match loaded.as_ref()? {
            Ok(models) => Ok(models
                .iter()
                .map(|info| {
                    let entry = ModelEntry::from_info(info);
                    let state = model_state(&entry, activity.get(&info.id), &gpu);
                    ModelRow { entry, state }
                })
                .collect()),
            Err(error) => Err(error.clone()),
        })
    }

    /// The state of the model with `id`; [`ModelState::NotDownloaded`] for an unknown id.
    #[must_use]
    pub fn state(&self, id: &str) -> ModelState {
        self.find(id).map_or(ModelState::NotDownloaded, |info| {
            model_state(
                &ModelEntry::from_info(&info),
                self.activity.read().get(id),
                &self.gpu.read(),
            )
        })
    }

    /// Whether a download is running.
    #[must_use]
    pub fn is_downloading(&self) -> bool {
        self.activity
            .read()
            .values()
            .any(|activity| matches!(activity, ModelActivity::Downloading(_)))
    }

    /// Whether a download or a check of `id` is running.
    #[must_use]
    pub fn is_busy(&self, id: &str) -> bool {
        matches!(
            self.activity.read().get(id),
            Some(ModelActivity::Downloading(_) | ModelActivity::Verifying)
        )
    }

    /// Starts downloading `id`; progress shows in [`ModelsHandle::state`] and a toast reports the end.
    pub fn download(self, id: &str) {
        if self.is_downloading() || self.is_busy(id) {
            return;
        }
        self.start(id, false);
    }

    /// Deletes the stored files of `id` and downloads them again, to repair a model that failed its check.
    pub fn download_again(self, id: &str) {
        if self.is_downloading() || self.is_busy(id) {
            return;
        }
        self.start(id, true);
    }

    fn start(mut self, id: &str, replace: bool) {
        let id = id.to_owned();
        self.activity.write().insert(
            id.clone(),
            ModelActivity::Downloading(crate::model_catalog::activity::DownloadProgress::default()),
        );
        let task = spawn_forever({
            let id = id.clone();
            async move { model_operations::download(self, id, replace).await }
        });
        self.tasks.write().insert(id, task);
    }

    /// Stops the download of `id` and removes what it had stored.
    pub fn cancel(mut self, id: &str) {
        let id = id.to_owned();
        if let Some(task) = self.tasks.write().remove(&id) {
            task.cancel();
        }
        self.activity.write().remove(&id);
        spawn_forever(async move {
            sleep(CANCEL_SETTLE).await;
            model_operations::discard(self, id).await;
        });
    }

    /// Hashes the stored files of `id` against the published checksums.
    pub fn verify(mut self, id: &str) {
        if self.is_busy(id) {
            return;
        }
        let id = id.to_owned();
        self.activity
            .write()
            .insert(id.clone(), ModelActivity::Verifying);
        spawn_forever(async move { model_operations::verify(self, id).await });
    }

    /// Deletes the stored files of `id`.
    pub fn remove(self, id: &str) {
        let id = id.to_owned();
        spawn_forever(async move { model_operations::remove(self, id).await });
    }

    /// Clears a failure or a repair prompt of `id` without acting on it.
    pub fn dismiss(mut self, id: &str) {
        let removable = matches!(
            self.activity.read().get(id),
            Some(ModelActivity::Failed(_) | ModelActivity::Damaged(_))
        );
        if removable {
            self.activity.write().remove(id);
        }
    }

    pub(super) fn set_activity(mut self, id: &str, activity: ModelActivity) {
        self.activity.write().insert(id.to_owned(), activity);
    }

    pub(super) fn clear_activity(mut self, id: &str) {
        self.activity.write().remove(id);
        self.tasks.write().remove(id);
    }

    pub(super) fn fail(mut self, id: &str, failure: DownloadFailure) {
        self.tasks.write().remove(id);
        self.set_activity(id, ModelActivity::Failed(failure));
    }
}

#[derive(Clone, Copy)]
struct Sources {
    models: Signal<Option<Result<Vec<ModelInfo>, String>>>,
    gpu: Signal<GpuSupport>,
    online: Signal<bool>,
    refresh: Signal<u32>,
    gpu_check: Signal<u32>,
    handle: ModelsHandle,
}

fn create_sources(toasts: Toasts) -> Sources {
    let models = Signal::new(None);
    let gpu = Signal::new(GpuSupport::Checking);
    let online = Signal::new(true);
    let refresh = Signal::new(0u32);
    let gpu_check = Signal::new(0u32);
    let handle = ModelsHandle {
        models: models.into(),
        gpu: gpu.into(),
        online: online.into(),
        activity: Signal::new(HashMap::new()),
        tasks: Signal::new(HashMap::new()),
        toasts,
        refresh,
        gpu_check,
    };
    Sources {
        models,
        gpu,
        online,
        refresh,
        gpu_check,
        handle,
    }
}

/// Loads the model list and watches the GPU and the connection; call once at the app root,
/// after the toast provider.
pub fn use_models_provider() -> ModelsHandle {
    let toasts = use_toasts();
    let Sources {
        mut models,
        mut gpu,
        mut online,
        refresh,
        gpu_check,
        handle,
    } = use_hook(|| Runtime::current().in_scope(ScopeId::ROOT, || create_sources(toasts)));

    use_resource(move || async move {
        gpu_check();
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

    use_future(move || async move {
        let mut watcher = document::eval(ONLINE_WATCHER);
        while let Ok(connected) = watcher.recv::<bool>().await {
            online.set(connected);
        }
    });

    use_resource(move || async move {
        refresh();
        let result = platform::get_platform()
            .list_available_models()
            .await
            .map_err(|error| error.to_string());
        models.set(Some(result));
    });

    use_context_provider(|| handle)
}

/// The models handle provided at the app root.
pub fn use_models() -> ModelsHandle {
    use_context::<ModelsHandle>()
}
