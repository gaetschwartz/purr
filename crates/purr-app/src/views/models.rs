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
        page::Page,
        spinner::Spinner,
    },
    hooks::use_models::{GpuSupport, ModelsHandle, use_models},
    model_catalog::{
        family::ModelGroup,
        listing::{Listing, ModelQuery, ModelRow, build_listing, summarize},
    },
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
            "This browser has no WebGPU",
            "Models can be downloaded, but none can run here.",
        )),
        GpuSupport::Available { shader_f16, .. } => (!shader_f16).then_some((
            BannerTone::Info,
            "Some models need shader-f16",
            "This GPU lacks it, so those models are greyed out.",
        )),
    }
}

#[component]
fn Catalog(models: ModelsHandle, rows: Vec<ModelRow>, selected: Option<String>) -> Element {
    let mut query = use_signal(ModelQuery::default);
    let mut delete_target = use_signal(|| None::<String>);

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
                "Downloads need a connection."
            }
        }
        if let Some((tone, title, body)) = notice {
            Banner { tone, title, "{body}" }
        }
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
            EmptyState { icon: IconName::Search, title: "No models match" }
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
                    "This frees {ByteSize(row.entry.size_bytes)}."
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
    let saved = use_memo(move || settings.with(|s| s.model.model_name.clone()));
    let selected = models
        .selection(&saved())
        .and_then(|selection| selection.row().map(|row| row.entry.id.clone()));
    let subtitle = models.rows().and_then(Result::ok).map(|rows| {
        let summary = summarize(&rows);
        format!(
            "{} of {} downloaded · {}",
            summary.installed,
            summary.total,
            ByteSize(summary.used_bytes)
        )
    });

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
                        title: "Couldn't list the models",
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
                    Catalog { models, rows, selected }
                },
            }
        }
    }
}
