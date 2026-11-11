use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonSize, ButtonVariant},
        icon::IconName,
        progress_bar::{Progress, ProgressBar},
    },
    hooks::use_models::{ModelsHandle, use_models},
    model_catalog::{
        copy,
        entry::ModelEntry,
        format::{rate_and_eta, transferred},
        state::{ModelState, UnavailableReason},
    },
    settings_store::use_settings,
    utils::byte_size::ByteSize,
};
use dioxus::prelude::*;

/// The selected model and where it stands; `None` until the list loaded or when it is unknown.
pub fn selected_model(models: ModelsHandle, id: &str) -> Option<(ModelEntry, ModelState)> {
    let info = models.find(id)?;
    Some((ModelEntry::from_info(&info), models.state(id)))
}

/// Whether the selected model can transcribe right now.
pub fn is_ready(selected: Option<&(ModelEntry, ModelState)>) -> bool {
    selected.is_some_and(|(_, state)| matches!(state, ModelState::Downloaded))
}

/// Explains, with a fix, why the selected model cannot transcribe yet; nothing when it can.
#[component]
pub fn ModelNotice() -> Element {
    let models = use_models();
    let settings = use_settings();
    let id = settings.with(|settings| settings.model.model_name.clone());

    if let Some(Err(error)) = models.rows() {
        return rsx! {
            Banner {
                tone: BannerTone::Bad,
                title: "Purr couldn't list the models",
                role: "alert",
                actions: rsx! {
                    Button {
                        size: ButtonSize::Small,
                        icon: IconName::Retry,
                        onclick: move |_| models.refresh(),
                        "Try again"
                    }
                },
                "{error}"
            }
        };
    }
    let Some((entry, state)) = selected_model(models, &id) else {
        if models.models.read().is_none() {
            return rsx! {};
        }
        return rsx! {
            Banner { tone: BannerTone::Warn, title: "The saved model isn't available here", role: "alert",
                "\"{id}\" can't run in this version of Purr. Pick another one with the Model button."
            }
        };
    };

    let download_id = entry.id.clone();
    match state {
        ModelState::Downloaded => rsx! {},
        ModelState::NotDownloaded => rsx! {
            Banner {
                tone: BannerTone::Warn,
                icon: IconName::Download,
                title: "{entry.label} isn't on this device yet",
                actions: rsx! {
                    Button {
                        variant: ButtonVariant::Primary,
                        size: ButtonSize::Small,
                        icon: IconName::Download,
                        disabled: models.is_downloading(),
                        onclick: move |_| models.download(&download_id),
                        "Download {ByteSize(entry.size_bytes)}"
                    }
                },
                "Download it once, or pick a model you already have with the Model button."
            }
        },
        ModelState::Downloading(progress) => rsx! {
            div { class: "card glass stack", role: "status", style: "gap:10px",
                b { "Downloading {entry.label}" }
                ProgressBar {
                    progress: progress.fraction().map_or(Progress::Indeterminate, Progress::Determinate),
                    label: "Download of {entry.label}",
                }
                div { class: "dlmeta",
                    span { {transferred(&progress)} }
                    span { {rate_and_eta(&progress)} }
                }
            }
        },
        ModelState::Verifying => rsx! {
            Banner { tone: BannerTone::Info, title: "Checking the files of {entry.label}…", role: "status" }
        },
        ModelState::Failed(failure) => {
            let text = copy::failure(&failure);
            rsx! {
                Banner {
                    tone: BannerTone::Bad,
                    title: "{text.title}",
                    role: "alert",
                    actions: rsx! {
                        Button {
                            variant: ButtonVariant::Primary,
                            size: ButtonSize::Small,
                            icon: IconName::Retry,
                            onclick: move |_| models.download(&download_id),
                            "Try again"
                        }
                    },
                    "{text.body}"
                }
            }
        }
        ModelState::NeedsRepair(damage) => rsx! {
            Banner {
                tone: BannerTone::Bad,
                title: "The files of {entry.label} are damaged",
                role: "alert",
                actions: rsx! {
                    Button {
                        variant: ButtonVariant::Primary,
                        size: ButtonSize::Small,
                        icon: IconName::Download,
                        onclick: move |_| models.download_again(&download_id),
                        "Download again"
                    }
                },
                {copy::damage(&damage)}
            }
        },
        ModelState::Unavailable { reason, .. } => {
            let why = match reason {
                UnavailableReason::NeedsShaderF16 => {
                    "It needs a GPU feature (shader-f16) this device lacks."
                }
                UnavailableReason::NeedsWebGpu => {
                    "It needs WebGPU, which this browser doesn't offer."
                }
            };
            rsx! {
                Banner { tone: BannerTone::Bad, title: "{entry.label} can't run here", role: "alert",
                    "{why} Pick another model with the Model button."
                }
            }
        }
    }
}
