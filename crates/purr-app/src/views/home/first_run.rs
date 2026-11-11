use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonSize, ButtonVariant},
        chip::{Chip, ChipTone},
        icon::{Icon, IconName, IconSize},
        model_meters::ModelMeters,
        page::Page,
        progress_bar::{Progress, ProgressBar},
        spinner::Spinner,
        toast::use_toasts,
    },
    hooks::use_models::{GpuSupport, use_models},
    main_app::Route,
    model_catalog::{
        copy,
        entry::ModelEntry,
        family::ModelGroup,
        format::{rate_and_eta, transferred},
        recommend::{Device, recommend},
        state::ModelState,
    },
    platform::IS_WEB,
    settings_store::use_settings,
    utils::byte_size::ByteSize,
};
use dioxus::prelude::*;

fn variant(entry: &ModelEntry) -> String {
    [
        Some(entry.language.label().to_owned()),
        entry.precision.tag(),
        Some(ByteSize(entry.size_bytes).to_string()),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(" · ")
}

/// No model on this device yet: the welcome card with a recommended download.
#[component]
pub fn FirstRun() -> Element {
    let models = use_models();
    let settings = use_settings();
    let toasts = use_toasts();
    let navigator = use_navigator();

    let gpu = models.gpu.read().clone();
    let device = if IS_WEB {
        Some(Device::Browser)
    } else {
        match gpu {
            GpuSupport::Checking => None,
            GpuSupport::Available { .. } => Some(Device::Desktop { gpu: true }),
            GpuSupport::Unavailable => Some(Device::Desktop { gpu: false }),
        }
    };
    let rows = match models.rows() {
        Some(Ok(rows)) => rows,
        Some(Err(error)) => {
            return rsx! {
                Page { title: "Welcome to Purr",
                    Banner { tone: BannerTone::Bad, title: "Purr couldn't list the models", role: "alert",
                        "{error}"
                    }
                }
            };
        }
        None => Vec::new(),
    };
    let recommendation = device.and_then(|device| recommend(&rows, device));
    let state_of = |entry: &ModelEntry| {
        rows.iter()
            .find(|row| row.entry.id == entry.id)
            .map_or(ModelState::NotDownloaded, |row| row.state.clone())
    };

    let download = move |entry: ModelEntry| {
        if let Err(error) = settings.set_model(entry.id.clone()) {
            toasts.error("That model can't be used", error.to_string());
            return;
        }
        models.download(&entry.id);
    };

    let (subtitle, gpu_line) = if IS_WEB {
        (
            "Private speech to text. Everything runs on this computer, in your browser.",
            "Uses your graphics card through WebGPU.",
        )
    } else {
        (
            "Private speech to text. Everything runs on this computer.",
            "Uses your graphics card when one is available, otherwise the CPU.",
        )
    };

    rsx! {
        Page { title: "Welcome to Purr", subtitle,
            section { class: "card glass welcome", aria_labelledby: "welcome-heading",
                div {
                    Chip { tone: ChipTone::Ok, icon: IconName::Lock, "On-device" }
                    h2 { id: "welcome-heading", "First, download a speech model." }
                    p {
                        "Purr turns audio into text with Whisper models that run on your own machine. You download a model once; after that, transcription works offline and your audio is never uploaded."
                    }
                    ul {
                        li {
                            Icon { name: IconName::Check, size: IconSize::Small }
                            span { "Models are stored on this device. You can add or remove them any time." }
                        }
                        li {
                            Icon { name: IconName::Check, size: IconSize::Small }
                            span { "{gpu_line}" }
                        }
                        li {
                            Icon { name: IconName::Check, size: IconSize::Small }
                            span { "99 languages, detected automatically." }
                        }
                    }
                }
                div { class: "reco", aria_live: "polite",
                    match recommendation {
                        None if device.is_none() || models.rows().is_none() => rsx! {
                            div { class: "row", role: "status",
                                Spinner {}
                                span { class: "muted", "Checking this device…" }
                            }
                        },
                        None => rsx! {
                            p { class: "muted", style: "margin:0",
                                "No model in the catalog can run on this device."
                            }
                        },
                        Some(found) => {
                            let model = found.model.clone();
                            let state = state_of(&model);
                            let smaller = found.smaller;
                            let pitch = if model.group == ModelGroup::Turbo {
                                "The best balance for most people: close to the most accurate model, several times faster."
                            } else {
                                "Quick to download and fast on any computer. Good for everyday recordings."
                            };
                            rsx! {
                                div { class: "top",
                                    div {
                                        Chip { tone: ChipTone::Accent, "Recommended" }
                                        div { class: "nm", style: "margin-top:8px", "{model.name}" }
                                        div { class: "faint", style: "font-size:13px", {variant(&model)} }
                                    }
                                }
                                p { class: "muted", style: "margin:0;font-size:14px", "{pitch}" }
                                if let Some(ratings) = model.group.ratings() {
                                    ModelMeters { ratings }
                                }
                                RecommendedAction { model, state, on_download: download }
                                div { class: "row wrap", style: "gap:4px;justify-content:space-between",
                                    if let Some(smaller) = smaller {
                                        Button {
                                            variant: ButtonVariant::Plain,
                                            size: ButtonSize::Small,
                                            disabled: models.is_downloading(),
                                            onclick: {
                                                let smaller = smaller.clone();
                                                move |_| download(smaller.clone())
                                            },
                                            "Smaller: {smaller.name} · {ByteSize(smaller.size_bytes)}"
                                        }
                                    }
                                    Button {
                                        variant: ButtonVariant::Plain,
                                        size: ButtonSize::Small,
                                        onclick: move |_| {
                                            navigator.push(Route::Models {});
                                        },
                                        "See all models"
                                        Icon { name: IconName::ChevronRight, size: IconSize::Small }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            p { class: "privacy-line",
                Icon { name: IconName::Lock, size: IconSize::Small }
                "The only network request is the model download itself."
            }
        }
    }
}

#[component]
fn RecommendedAction(
    model: ModelEntry,
    state: ModelState,
    on_download: EventHandler<ModelEntry>,
) -> Element {
    let models = use_models();
    let id = model.id.clone();
    match state {
        ModelState::Downloading(progress) => rsx! {
            div { class: "dl", role: "status",
                ProgressBar {
                    progress: progress.fraction().map_or(Progress::Indeterminate, Progress::Determinate),
                    label: "Download of {model.label}",
                }
                div { class: "dlmeta",
                    span { {transferred(&progress)} }
                    span { {rate_and_eta(&progress)} }
                }
                div { class: "row", style: "justify-content:flex-end",
                    Button {
                        size: ButtonSize::Small,
                        onclick: move |_| models.cancel(&id),
                        "Cancel download"
                    }
                }
            }
        },
        ModelState::Verifying => rsx! {
            div { class: "row", role: "status",
                Spinner {}
                span { class: "muted", "Checking the downloaded files…" }
            }
        },
        ModelState::Failed(failure) => {
            let text = copy::failure(&failure);
            rsx! {
                div { class: "banner bad", role: "alert",
                    Icon { name: IconName::XCircle }
                    div {
                        b { "{text.title}" }
                        p { "{text.body}" }
                        div { class: "acts",
                            Button {
                                variant: ButtonVariant::Primary,
                                size: ButtonSize::Small,
                                icon: IconName::Retry,
                                onclick: move |_| on_download.call(model.clone()),
                                "Try again"
                            }
                        }
                    }
                }
            }
        }
        ModelState::NotDownloaded
        | ModelState::Downloaded
        | ModelState::NeedsRepair(_)
        | ModelState::Unavailable { .. } => rsx! {
            Button {
                variant: ButtonVariant::Primary,
                size: ButtonSize::Large,
                icon: IconName::Download,
                disabled: models.is_downloading(),
                onclick: {
                    let model = model.clone();
                    move |_| on_download.call(model.clone())
                },
                "Download {model.name} · {ByteSize(model.size_bytes)}"
            }
        },
    }
}
