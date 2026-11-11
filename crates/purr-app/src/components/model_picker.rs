use super::{Button, ButtonSize, ButtonVariant, ProgressBar, Select, SelectOption};
use crate::{
    hooks::use_models::{GpuSupport, ModelsHandle},
    platform,
    utils::byte_size::ByteSize,
};
use dioxus::prelude::*;
use futures::StreamExt as _;
use purr_common::platform::{ModelInfo, ModelOperationProgress, Platform as _, model_flag};

/// Where a model download stands.
#[derive(Debug, Clone, PartialEq)]
enum DownloadState {
    Idle,
    Starting,
    Downloading {
        loaded: u64,
        total: Option<u64>,
        speed_bps: Option<u64>,
    },
    Verifying,
    Failed(String),
}

/// Whether this build can run `model` on the GPU it found.
fn is_runnable(model: &ModelInfo, gpu: &GpuSupport) -> bool {
    let needs_f16 = model.metadata.has_flag(model_flag::REQUIRES_SHADER_F16);
    match gpu {
        GpuSupport::Available { shader_f16, .. } => *shader_f16 || !needs_f16,
        GpuSupport::Checking | GpuSupport::Unavailable => true,
    }
}

fn option_label(model: &ModelInfo, runnable: bool) -> String {
    let mut label = format!("{} ({})", model.name, ByteSize(model.size_bytes));
    if model.is_installed {
        label.push_str(" - downloaded");
    }
    if !runnable {
        label.push_str(" - needs WebGPU shader-f16");
    }
    label
}

/// Model selection with download management.
#[component]
pub fn ModelPicker(
    selected: String,
    models: ModelsHandle,
    onselect: EventHandler<String>,
) -> Element {
    let mut download = use_signal(|| DownloadState::Idle);
    let mut task = use_signal(|| None::<dioxus::core::Task>);
    let mut remove_error = use_signal(|| None::<String>);
    let mut verified = use_signal(|| false);
    let gpu = models.gpu.read().clone();

    let list = models.models.read().clone();
    let Some(list) = list else {
        return rsx! {
            p { class: "text-sm text-gray-500 dark:text-gray-400", role: "status", "Loading models..." }
        };
    };
    let list = match list {
        Ok(list) => list,
        Err(error) => {
            return rsx! {
                p { class: "text-sm text-red-600 dark:text-red-400", role: "alert", "Could not list the models: {error}" }
            };
        }
    };

    let current = list.iter().find(|model| model.id == selected).cloned();
    let mut options: Vec<SelectOption> = list
        .iter()
        .map(|model| {
            let runnable = is_runnable(model, &gpu);
            SelectOption {
                value: model.id.clone(),
                label: option_label(model, runnable),
                disabled: !runnable,
            }
        })
        .collect();
    if current.is_none() {
        options.insert(
            0,
            SelectOption {
                value: selected.clone(),
                label: format!("{selected} (not available here)"),
                disabled: true,
            },
        );
    }

    let download_model = {
        let id = selected.clone();
        move |_| {
            let id = id.clone();
            download.set(DownloadState::Starting);
            verified.set(false);
            let handle = spawn(async move {
                match platform::get_platform().fetch_model(&id).await {
                    Ok(mut stream) => {
                        while let Some(item) = stream.next().await {
                            match item {
                                Ok(ModelOperationProgress::Starting { .. }) => {}
                                Ok(ModelOperationProgress::Downloading {
                                    bytes_downloaded,
                                    total_bytes,
                                    speed_bps,
                                    ..
                                }) => download.set(DownloadState::Downloading {
                                    loaded: bytes_downloaded,
                                    total: total_bytes,
                                    speed_bps,
                                }),
                                Ok(ModelOperationProgress::Installing { .. }) => {
                                    download.set(DownloadState::Verifying);
                                }
                                Ok(ModelOperationProgress::Completed { .. }) => {
                                    verified.set(true);
                                    download.set(DownloadState::Idle);
                                    models.refresh();
                                    break;
                                }
                                Ok(ModelOperationProgress::Failed { error_message, .. }) => {
                                    download.set(DownloadState::Failed(error_message));
                                    models.refresh();
                                    break;
                                }
                                Err(error) => {
                                    download.set(DownloadState::Failed(error.to_string()));
                                    break;
                                }
                            }
                        }
                    }
                    Err(error) => download.set(DownloadState::Failed(error.to_string())),
                }
                task.set(None);
            });
            task.set(Some(handle));
        }
    };

    let cancel_download = move |_| {
        if let Some(handle) = task.take() {
            handle.cancel();
        }
        download.set(DownloadState::Idle);
    };

    let remove_model = {
        let id = selected.clone();
        move |_| {
            let id = id.clone();
            spawn(async move {
                match platform::get_platform().remove_model(&id).await {
                    Ok(()) => remove_error.set(None),
                    Err(error) => remove_error.set(Some(error.to_string())),
                }
                models.refresh();
            });
        }
    };

    let state = download.read().clone();
    let busy = !matches!(state, DownloadState::Idle | DownloadState::Failed(_));

    rsx! {
        Select {
            label: "Model",
            description: "Larger models are more accurate but slower and need a bigger download",
            value: selected.clone(),
            options,
            onchange: move |value: String| onselect.call(value),
        }

        if let Some(model) = current {
            div { class: "space-y-3 rounded-lg border border-gray-200 dark:border-gray-700 p-4",
                p { class: "text-sm text-gray-600 dark:text-gray-300", "{model.description}" }

                match state {
                    DownloadState::Idle | DownloadState::Failed(_) => rsx! {
                        div { class: "flex flex-wrap items-center gap-3",
                            if model.is_installed {
                                span { class: "text-sm font-medium text-green-700 dark:text-green-400", "Downloaded and ready" }
                                Button {
                                    variant: ButtonVariant::Secondary,
                                    size: ButtonSize::Small,
                                    onclick: download_model,
                                    "Verify files"
                                }
                                Button {
                                    variant: ButtonVariant::Secondary,
                                    size: ButtonSize::Small,
                                    onclick: remove_model,
                                    "Remove download"
                                }
                            } else {
                                span { class: "text-sm text-gray-600 dark:text-gray-300",
                                    {format!("Not downloaded yet ({})", ByteSize(model.size_bytes))}
                                }
                                Button {
                                    size: ButtonSize::Small,
                                    onclick: download_model,
                                    "Download now"
                                }
                            }
                        }
                    },
                    DownloadState::Starting => rsx! {
                        p { class: "text-sm text-gray-600 dark:text-gray-300", role: "status", "Starting..." }
                    },
                    DownloadState::Downloading { loaded, total, speed_bps } => {
                        let total = total.unwrap_or(model.size_bytes).max(1);
                        let percent = loaded as f32 / total as f32 * 100.0;
                        let speed = speed_bps.map(|bps| format!(" at {}/s", ByteSize(bps))).unwrap_or_default();
                        rsx! {
                            div { class: "space-y-2", role: "status",
                                ProgressBar {
                                    value: percent,
                                    label: format!("Downloading {} of {}{speed}", ByteSize(loaded), ByteSize(total)),
                                }
                            }
                        }
                    }
                    DownloadState::Verifying => rsx! {
                        p { class: "text-sm text-gray-600 dark:text-gray-300", role: "status", "Checking the downloaded files..." }
                    },
                }

                if busy {
                    Button {
                        variant: ButtonVariant::Secondary,
                        size: ButtonSize::Small,
                        onclick: cancel_download,
                        "Cancel download"
                    }
                }

                if let DownloadState::Failed(message) = download.read().clone() {
                    p { class: "text-sm text-red-600 dark:text-red-400", role: "alert", "The model could not be installed: {message}" }
                }
                if verified() && model.is_installed {
                    p { class: "text-sm text-green-700 dark:text-green-400", role: "status",
                        "All files match their published checksums."
                    }
                }
                if let Some(message) = remove_error() {
                    p { class: "text-sm text-red-600 dark:text-red-400", role: "alert", "Could not remove the model: {message}" }
                }
            }
        } else {
            p { class: "text-sm text-red-600 dark:text-red-400", role: "alert",
                "The saved model \"{selected}\" is not available here. Choose another model and save."
            }
        }
    }
}
