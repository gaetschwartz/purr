use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonVariant},
        dialog::{Dialog, DialogTone},
        empty_state::EmptyState,
        icon::IconName,
        model_filters::ModelFilters,
        model_row::ModelListRow,
        model_section::ModelSection,
        model_stats::{Headline, ModelStats},
        page::Page,
        spinner::Spinner,
    },
    hooks::use_models::{GpuSupport, ModelsHandle, use_models},
    model_catalog::{
        entry::Precision,
        family::ModelGroup,
        format,
        listing::{Listing, ModelQuery, ModelRow, active_download, build_listing, summarize},
    },
    platform::IS_WEB,
    settings_store::use_settings,
    utils::byte_size::ByteSize,
};
use dioxus::prelude::*;

const IN_PROGRESS: &str = "In progress";

fn gpu_notice(
    gpu: &GpuSupport,
    rows: &[ModelRow],
) -> Option<(BannerTone, &'static str, &'static str)> {
    if !rows.iter().any(|row| row.entry.requires_shader_f16) {
        return None;
    }
    match gpu {
        GpuSupport::Checking => None,
        GpuSupport::Unavailable => Some((
            BannerTone::Warn,
            "WebGPU isn't available in this browser",
            "Purr runs browser models on your graphics card through WebGPU. You can download models, but none can run until you use a browser that offers it, such as a current Chrome or Edge.",
        )),
        GpuSupport::Available { shader_f16, .. } => (!shader_f16).then_some((
            BannerTone::Info,
            "Some models need a GPU feature this device doesn't offer",
            "The largest model uses 16-bit floating point on the GPU (WebGPU \"shader-f16\"). Your GPU doesn't expose it here, so pick one of the smaller models instead.",
        )),
    }
}

fn headline(rows: &[ModelRow], selected: Option<&str>) -> Headline {
    if let Some(row) = active_download(rows) {
        return Headline::Downloading {
            label: row.entry.label.clone(),
            percent: row.state.progress().and_then(format::percent),
        };
    }
    let label = selected.map_or_else(
        || "Not chosen yet".to_owned(),
        |id| {
            rows.iter()
                .find(|row| row.entry.id == id)
                .map_or_else(|| id.to_owned(), |row| row.entry.label.clone())
        },
    );
    Headline::InUse(label)
}

#[component]
fn Catalog(models: ModelsHandle, rows: Vec<ModelRow>, selected: Option<String>) -> Element {
    let mut query = use_signal(ModelQuery::default);
    let mut delete_target = use_signal(|| None::<String>);

    let has_q5 = rows
        .iter()
        .any(|row| row.entry.precision == Precision::Quantized { bits: 5 });
    let summary = summarize(&rows);
    let headline = headline(&rows, selected.as_deref());
    let groups: Vec<ModelGroup> = ModelGroup::ORDER
        .into_iter()
        .filter(|group| rows.iter().any(|row| row.entry.group == *group))
        .collect();
    let notice = gpu_notice(&models.gpu.read(), &rows);
    let offline = !*models.online.read();
    let Listing {
        in_progress,
        groups: sections,
    } = build_listing(rows.clone(), &query.read());
    let nothing_matches =
        in_progress.is_empty() && sections.iter().all(|section| section.rows.is_empty());
    let in_use = |id: &str| selected.as_deref() == Some(id);

    rsx! {
        if offline {
            Banner {
                tone: BannerTone::Warn,
                icon: IconName::WifiOff,
                title: "You're offline",
                role: "status",
                "Downloads need a connection. Models you've already downloaded keep working."
            }
        }
        if let Some((tone, title, body)) = notice {
            Banner { tone, title, "{body}" }
        }
        ModelStats { summary, headline }
        ModelFilters {
            query: query(),
            groups,
            onchange: move |next| query.set(next),
        }
        if !in_progress.is_empty() {
            ModelSection { title: IN_PROGRESS,
                for row in in_progress {
                    ModelListRow {
                        key: "{row.entry.id}",
                        in_use: in_use(&row.entry.id),
                        row,
                        on_delete: move |id| delete_target.set(Some(id)),
                    }
                }
            }
        }
        if nothing_matches {
            EmptyState { icon: IconName::Search, title: "No models match",
                "Try a different filter or search."
            }
        }
        for section in sections {
            ModelSection {
                key: "{section.group.label()}",
                title: section.group.label(),
                hint: section.group.hint(),
                for row in section.rows {
                    ModelListRow {
                        key: "{row.entry.id}",
                        in_use: in_use(&row.entry.id),
                        row,
                        on_delete: move |id| delete_target.set(Some(id)),
                    }
                }
            }
        }
        p { class: "foot", style: "padding-left:16px",
            "Models download over HTTPS and are checked against a published checksum."
            if has_q5 {
                " Q5 means quantized to 5-bit: about a third of the size, with a small loss in accuracy."
            }
        }
        if let Some(target) = delete_target() {
            if let Some(row) = rows.iter().find(|row| row.entry.id == target) {
                Dialog {
                    title: "Delete {row.entry.label}?",
                    confirm_label: "Delete model",
                    tone: DialogTone::Destructive,
                    on_cancel: move |()| delete_target.set(None),
                    on_confirm: {
                        let id = target.clone();
                        move |()| {
                            delete_target.set(None);
                            models.remove(&id);
                        }
                    },
                    "This frees {ByteSize(row.entry.size_bytes)}. You can download it again later."
                    if in_use(&row.entry.id) {
                        " It is the model Purr uses for transcription, so choose another one before your next file."
                    }
                }
            }
        }
    }
}

/// The model manager: what is stored, what can be downloaded, and the actions on each.
#[component]
pub fn Models() -> Element {
    let models = use_models();
    let settings = use_settings();
    let selected_id = use_memo(move || settings.with(|s| s.model.model_name.clone()));
    let subtitle = if IS_WEB {
        "Speech models stored in this browser. Larger models are more accurate and slower."
    } else {
        "Speech models stored on this device. Larger models are more accurate and slower."
    };

    rsx! {
        Page { title: "Models", subtitle, wide: true,
            match models.rows() {
                None => rsx! {
                    div { class: "row", role: "status",
                        Spinner {}
                        span { class: "muted", "Loading models…" }
                    }
                },
                Some(Err(error)) => rsx! {
                    Banner {
                        tone: BannerTone::Bad,
                        title: "Could not list the models",
                        role: "alert",
                        actions: rsx! {
                            Button {
                                variant: ButtonVariant::Primary,
                                icon: IconName::Retry,
                                onclick: move |_| models.refresh(),
                                "Try again"
                            }
                        },
                        "{error}"
                    }
                },
                Some(Ok(rows)) => rsx! {
                    Catalog { models, rows, selected: Some(selected_id()) }
                },
            }
        }
    }
}
