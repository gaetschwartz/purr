use super::{
    banner::{Banner, BannerTone},
    button::{Button, ButtonSize, ButtonVariant},
    chip::{Chip, ChipTone},
    icon::IconName,
    progress_bar::{Progress, ProgressBar},
    select::{Select, SelectOption},
    spinner::Spinner,
};
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
            div { class: "row", role: "status",
                Spinner {}
                span { class: "muted", "Loading models…" }
            }
        };
    };
    let list = match list {
        Ok(list) => list,
        Err(error) => {
            return rsx! {
                Banner { tone: BannerTone::Bad, title: "Could not list the models", role: "alert", "{error}" }
            };
        }
    };

    let current = list.iter().find(|model| model.id == selected).cloned();
    let mut options: Vec<SelectOption<String>> = list
        .iter()
        .map(|model| {
            let runnable = is_runnable(model, &gpu);
            SelectOption::new(model.id.clone(), option_label(model, runnable)).disabled(!runnable)
        })
        .collect();
    if current.is_none() {
        options.insert(
            0,
            SelectOption::new(selected.clone(), format!("{selected} (not available here)"))
                .disabled(true),
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
        div { class: "stack",
            div { class: "row wrap",
                span { class: "grow", id: "model-select-label", b { "Model" } }
                Select {
                    labelled_by: "model-select-label",
                    value: selected.clone(),
                    options,
                    onchange: move |value: String| onselect.call(value),
                }
            }

            if let Some(model) = current {
                p { class: "muted", style: "margin:0", "{model.description}" }

                match state {
                    DownloadState::Idle | DownloadState::Failed(_) => rsx! {
                        div { class: "row wrap",
                            if model.is_installed {
                                Chip { tone: ChipTone::Ok, icon: IconName::CheckCircle, "Downloaded and ready" }
                                Button {
                                    size: ButtonSize::Small,
                                    icon: IconName::Shield,
                                    onclick: download_model,
                                    "Verify files"
                                }
                                Button {
                                    variant: ButtonVariant::Danger,
                                    size: ButtonSize::Small,
                                    icon: IconName::Trash,
                                    onclick: remove_model,
                                    "Remove download"
                                }
                            } else {
                                span { class: "muted", {format!("Not downloaded yet ({})", ByteSize(model.size_bytes))} }
                                Button {
                                    variant: ButtonVariant::Primary,
                                    size: ButtonSize::Small,
                                    icon: IconName::Download,
                                    onclick: download_model,
                                    "Download now"
                                }
                            }
                        }
                    },
                    DownloadState::Starting => rsx! {
                        div { class: "row", role: "status",
                            Spinner {}
                            span { class: "muted", "Starting…" }
                        }
                    },
                    DownloadState::Downloading { loaded, total, speed_bps } => {
                        let total = total.unwrap_or(model.size_bytes).max(1);
                        let fraction = loaded as f64 / total as f64;
                        let speed = speed_bps.map(|bps| format!("{}/s", ByteSize(bps))).unwrap_or_default();
                        rsx! {
                            div { class: "dl", role: "status",
                                ProgressBar {
                                    progress: Progress::Determinate(fraction),
                                    label: "Model download",
                                }
                                div { class: "dlmeta",
                                    span { {format!("{} of {}", ByteSize(loaded), ByteSize(total))} }
                                    span { "{speed}" }
                                }
                            }
                        }
                    }
                    DownloadState::Verifying => rsx! {
                        div { class: "row", role: "status",
                            Spinner {}
                            span { class: "muted", "Checking the downloaded files…" }
                        }
                    },
                }

                if busy {
                    div { class: "row",
                        Button { size: ButtonSize::Small, onclick: cancel_download, "Cancel download" }
                    }
                }

                if let DownloadState::Failed(message) = download.read().clone() {
                    Banner { tone: BannerTone::Bad, title: "The model could not be installed", role: "alert", "{message}" }
                }
                if verified() && model.is_installed {
                    Banner { tone: BannerTone::Ok, title: "All files match their published checksums", role: "status" }
                }
                if let Some(message) = remove_error() {
                    Banner { tone: BannerTone::Bad, title: "Could not remove the model", role: "alert", "{message}" }
                }
            } else {
                Banner { tone: BannerTone::Warn, title: "The saved model is not available here", role: "alert",
                    "\"{selected}\" cannot run in this app. Choose another model and save."
                }
            }
        }
    }
}
