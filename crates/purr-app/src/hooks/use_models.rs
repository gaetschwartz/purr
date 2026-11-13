use crate::{
    components::toast::{Toasts, use_toasts},
    hooks::model_operations,
    model_catalog::{
        activity::{DownloadFailure, DownloadProgress, ModelActivity, VerifyProgress},
        entry::ModelEntry,
        listing::ModelRow,
        recommend::Device,
        selection::{Selection, resolve},
        state::{ModelState, model_state},
    },
    platform::{self, IS_WEB},
    settings_store::{SettingsStore, use_settings},
    utils::delay::sleep,
};
use dioxus::{
    core::{Runtime, ScopeId, Task, spawn_forever},
    prelude::*,
};
use futures::channel::oneshot;
use purr_common::accel::AccelPolicy;
use purr_common::platform::{DeviceType, ModelInfo, Platform as _};
use std::{collections::HashMap, time::Duration};

/// How a download someone waits on ended; a cancelled download drops the waiters instead.
pub type DownloadEnd = Result<(), DownloadFailure>;

struct RunningDownload {
    task: Task,
    waiters: Vec<oneshot::Sender<DownloadEnd>>,
}

/// What the GPU can do for transcription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuSupport {
    Checking,
    Unavailable,
    Available {
        name: Option<String>,
        shader_f16: bool,
    },
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
    downloads: Signal<HashMap<String, RunningDownload>>,
    pub(super) toasts: Toasts,
    pub(super) settings: SettingsStore,
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

    /// Whether the settings keep browser runs on the CPU (subscribes the caller).
    fn cpu_chosen(&self) -> bool {
        IS_WEB
            && self
                .settings
                .with(|settings| settings.performance.accel_policy())
                == AccelPolicy::CPU_ONLY
    }

    /// The catalog with a state per model (subscribes the caller); `None` until the list has loaded.
    #[must_use]
    pub fn rows(&self) -> Option<Result<Vec<ModelRow>, String>> {
        let gpu = self.gpu.read();
        let cpu_chosen = self.cpu_chosen();
        let activity = self.activity.read();
        let loaded = self.models.read();
        Some(match loaded.as_ref()? {
            Ok(models) => Ok(models
                .iter()
                .map(|info| {
                    let entry = ModelEntry::from_info(info);
                    let state = model_state(&entry, activity.get(&info.id), &gpu, cpu_chosen);
                    ModelRow { entry, state }
                })
                .collect()),
            Err(error) => Err(error.clone()),
        })
    }

    /// The model new transcriptions use for the saved id `saved` (subscribes the caller);
    /// `None` until the list has loaded, [`Selection::Missing`] when it failed to load.
    #[must_use]
    pub fn selection(&self, saved: &str) -> Option<Selection> {
        let device = if IS_WEB {
            Device::Browser {
                webgpu: *self.gpu.read() != GpuSupport::Unavailable,
            }
        } else {
            Device::Desktop {
                gpu: self.gpu.read().found(),
            }
        };
        Some(
            self.rows()?
                .map_or(Selection::Missing, |rows| resolve(&rows, saved, device)),
        )
    }

    /// The state of the model with `id`; [`ModelState::NotDownloaded`] for an unknown id.
    #[must_use]
    pub fn state(&self, id: &str) -> ModelState {
        self.find(id).map_or(ModelState::NotDownloaded, |info| {
            model_state(
                &ModelEntry::from_info(&info),
                self.activity.read().get(id),
                &self.gpu.read(),
                self.cpu_chosen(),
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
            Some(ModelActivity::Downloading(_) | ModelActivity::Verifying(_))
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
            ModelActivity::Downloading(DownloadProgress::default()),
        );
        let task = spawn_forever({
            let id = id.clone();
            async move { model_operations::download(self, id, replace).await }
        });
        self.downloads.write().insert(
            id,
            RunningDownload {
                task,
                waiters: Vec::new(),
            },
        );
    }

    /// Downloads `id`, or joins its running download, and resolves when it ends; `replace`
    /// deletes the stored files first. `None` while another model downloads.
    ///
    /// The receiver is cancelled with the download. While someone waits, the end of the
    /// download is theirs to report, so no toast says it.
    pub fn download_and_wait(
        self,
        id: &str,
        replace: bool,
    ) -> Option<oneshot::Receiver<DownloadEnd>> {
        let joining = matches!(
            self.activity.peek().get(id),
            Some(ModelActivity::Downloading(_))
        );
        if !joining {
            if self.is_downloading() || self.is_busy(id) {
                return None;
            }
            self.start(id, replace);
        }
        Some(self.when_downloaded(id))
    }

    fn when_downloaded(mut self, id: &str) -> oneshot::Receiver<DownloadEnd> {
        let (sender, receiver) = oneshot::channel();
        if let Some(download) = self.downloads.write().get_mut(id) {
            download.waiters.push(sender);
        }
        receiver
    }

    pub(super) fn has_waiters(&self, id: &str) -> bool {
        self.downloads
            .peek()
            .get(id)
            .is_some_and(|download| !download.waiters.is_empty())
    }

    pub(super) fn settle(mut self, id: &str, end: &DownloadEnd) {
        let waiting = self
            .downloads
            .write()
            .get_mut(id)
            .map(|download| std::mem::take(&mut download.waiters))
            .unwrap_or_default();
        for sender in waiting {
            if sender.send(end.clone()).is_err() {
                tracing::debug!("Nobody waits on the download of {id} any more");
            }
        }
    }

    /// Stops the download of `id` and removes what it had stored.
    pub fn cancel(mut self, id: &str) {
        let id = id.to_owned();
        if let Some(download) = self.downloads.write().remove(&id) {
            download.task.cancel();
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
        self.activity.write().insert(
            id.clone(),
            ModelActivity::Verifying(VerifyProgress::default()),
        );
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
        self.downloads.write().remove(id);
    }

    pub(super) fn fail(mut self, id: &str, failure: DownloadFailure) {
        self.downloads.write().remove(id);
        self.set_activity(id, ModelActivity::Failed(failure));
    }
}

#[derive(Clone, Copy)]
struct Sources {
    models: Signal<Option<Result<Vec<ModelInfo>, String>>>,
    gpu: Signal<GpuSupport>,
    online: Signal<bool>,
    refresh: Signal<u32>,
    handle: ModelsHandle,
}

fn create_sources(toasts: Toasts, settings: SettingsStore) -> Sources {
    let models = Signal::new(None);
    let gpu = Signal::new(GpuSupport::Checking);
    let online = Signal::new(true);
    let refresh = Signal::new(0u32);
    let handle = ModelsHandle {
        models: models.into(),
        gpu: gpu.into(),
        online: online.into(),
        activity: Signal::new(HashMap::new()),
        downloads: Signal::new(HashMap::new()),
        toasts,
        settings,
        refresh,
    };
    Sources {
        models,
        gpu,
        online,
        refresh,
        handle,
    }
}

/// Loads the model list and watches the GPU and the connection; call once at the app root,
/// after the toast and settings providers.
pub fn use_models_provider() -> ModelsHandle {
    let toasts = use_toasts();
    let settings = use_settings();
    let Sources {
        mut models,
        mut gpu,
        mut online,
        refresh,
        handle,
    } = use_hook(|| {
        Runtime::current().in_scope(ScopeId::ROOT, || create_sources(toasts, settings))
    });

    use_resource(move || async move {
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
                    name: Some(device.name).filter(|name| !name.trim().is_empty()),
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
