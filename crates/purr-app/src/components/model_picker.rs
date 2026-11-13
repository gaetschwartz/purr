use super::{
    icon::{Icon, IconName, IconSize},
    pill::Pill,
    popover::{OptionRole, Popover, PopoverOption},
    spinner::Spinner,
};
use crate::{
    components::toast::use_toasts,
    hooks::use_models::{GpuSupport, use_models},
    main_app::Route,
    model_catalog::{
        copy, entry::ModelEntry, family::ModelGroup, format::percent, listing::ModelRow,
        state::ModelState,
    },
    platform::IS_WEB,
    settings_store::use_settings,
    utils::byte_size::ByteSize,
};
use dioxus::prelude::*;
use purr_common::accel::AccelPolicy;

/// Language and precision, such as "Multilingual · Q5".
fn variant(entry: &ModelEntry) -> String {
    [
        Some(entry.language.label().to_owned()),
        entry.precision.tag(),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(" · ")
}

fn subtitle(row: &ModelRow) -> String {
    match &row.state {
        ModelState::Unavailable { reason, .. } => copy::unavailable(*reason).1.to_owned(),
        ModelState::Downloading(_) => "Downloading…".to_owned(),
        ModelState::Verifying => "Checking files…".to_owned(),
        ModelState::Failed(_) => "Last download failed".to_owned(),
        ModelState::NeedsRepair(_) => "Files damaged".to_owned(),
        ModelState::NotDownloaded | ModelState::Downloaded => variant(&row.entry),
    }
}

/// The size of a model that would have to be downloaded, or how far its download is.
fn download_size(row: &ModelRow) -> Option<String> {
    match &row.state {
        ModelState::Downloading(progress) => {
            Some(percent(progress).map_or_else(|| "…".to_owned(), |done| format!("{done}%")))
        }
        ModelState::Downloaded
        | ModelState::Unavailable {
            installed: true, ..
        } => None,
        ModelState::NotDownloaded
        | ModelState::Verifying
        | ModelState::Failed(_)
        | ModelState::NeedsRepair(_)
        | ModelState::Unavailable {
            installed: false, ..
        } => Some(ByteSize(row.entry.size_bytes).to_string()),
    }
}

/// Rows of `group`, smallest first.
fn in_group(rows: &[ModelRow], group: ModelGroup) -> Vec<ModelRow> {
    let mut found: Vec<ModelRow> = rows
        .iter()
        .filter(|row| row.entry.group == group)
        .cloned()
        .collect();
    found.sort_by_key(|row| row.entry.size_bytes);
    found
}

/// Every model of `use_models()` in a popover (a sheet on phones), anchored on the element `anchor`.
///
/// Render it only while open. Picking a model that can run here saves it with `set_model` and
/// calls `on_pick`, whether or not it is downloaded; a run downloads it first. Models that would
/// need a download show their size.
#[component]
pub fn ModelPicker(
    /// The id of the element the popover hangs from.
    #[props(into)]
    anchor: String,
    /// The id of the selected model.
    #[props(into)]
    selected: String,
    on_close: EventHandler<()>,
    /// Called after a model was picked and saved.
    on_pick: Option<EventHandler<ModelEntry>>,
) -> Element {
    let models = use_models();
    let settings = use_settings();
    let toasts = use_toasts();
    let navigator = use_navigator();

    let pick = move |row: ModelRow| {
        if matches!(row.state, ModelState::Unavailable { .. }) {
            return;
        }
        match settings.set_model(row.entry.id.clone()) {
            Ok(()) => {
                on_close.call(());
                if let Some(handler) = on_pick {
                    handler.call(row.entry);
                }
            }
            Err(error) => {
                toasts.error("That model can't be used", error.to_string());
            }
        }
    };

    let body = match models.rows() {
        None => rsx! {
            div { class: "row", role: "status", style: "padding:12px",
                Spinner {}
                span { class: "muted", "Loading models…" }
            }
        },
        Some(Err(error)) => rsx! {
            p { class: "muted", role: "alert", style: "padding:10px;margin:0",
                "Couldn't list the models: {error}"
            }
        },
        Some(Ok(rows)) => {
            let groups = ModelGroup::ORDER
                .into_iter()
                .map(|group| (group, in_group(&rows, group)))
                .filter(|(_, rows)| !rows.is_empty());
            rsx! {
                for (group , rows) in groups {
                    div {
                        class: "plist",
                        role: "listbox",
                        aria_label: "{group.label()}",
                        key: "{group.label()}",
                        div { class: "ph", "{group.label()}" }
                        for row in rows {
                            ModelOption {
                                key: "{row.entry.id}",
                                row: row.clone(),
                                selected: row.entry.id == selected,
                                onselect: pick,
                            }
                        }
                    }
                }
            }
        }
    };

    let on_cpu = IS_WEB
        && (*models.gpu.read() == GpuSupport::Unavailable
            || settings.with(|settings| settings.performance.accel_policy())
                == AccelPolicy::CPU_ONLY);

    rsx! {
        Popover { anchor, label: "Choose a model", on_close,
            if on_cpu {
                p { class: "foot", role: "note", style: "padding:8px 12px;margin:0",
                    "Models run on the CPU here. Smaller ones are faster."
                }
            }
            {body}
            hr {}
            div { class: "plist", role: "menu", aria_label: "Models",
                PopoverOption {
                    title: "Manage models…",
                    leading: rsx! {
                        Icon { name: IconName::Cube, size: IconSize::Small }
                    },
                    onselect: move |()| {
                        on_close.call(());
                        navigator.push(Route::Models {});
                    },
                }
            }
        }
    }
}

#[component]
fn ModelOption(row: ModelRow, selected: bool, onselect: EventHandler<ModelRow>) -> Element {
    let disabled = matches!(row.state, ModelState::Unavailable { .. });
    let size = download_size(&row);
    let prefix = match row.state {
        ModelState::Downloading(_) => "downloaded",
        ModelState::NotDownloaded
        | ModelState::Downloaded
        | ModelState::Verifying
        | ModelState::Failed(_)
        | ModelState::NeedsRepair(_)
        | ModelState::Unavailable { .. } => "download size",
    };
    rsx! {
        PopoverOption {
            title: row.entry.label.clone(),
            subtitle: subtitle(&row),
            trailing: size.clone(),
            trailing_prefix: size.map(|_| prefix.to_owned()),
            role: OptionRole::Option,
            selected,
            disabled,
            leading: rsx! {
                if selected {
                    Icon { name: IconName::Check, size: IconSize::Small }
                }
            },
            onselect: move |()| onselect.call(row.clone()),
        }
    }
}

/// The run-bar pill showing the model new transcriptions use; it opens a [`ModelPicker`].
///
/// A model that is not downloaded yet shows its download size.
#[component]
pub fn ModelPill(
    /// The pill's DOM id, which the popover hangs from.
    #[props(into)]
    id: String,
    #[props(default)] disabled: bool,
    /// Called after a model was picked and saved.
    on_pick: Option<EventHandler<ModelEntry>>,
) -> Element {
    let models = use_models();
    let settings = use_settings();
    let mut open = use_signal(|| false);
    let saved = settings.with(|settings| settings.model.model_name.clone());
    let selection = models.selection(&saved);
    let row = selection
        .as_ref()
        .and_then(|selection| selection.row().cloned());
    let (value, detail, selected) = match (&selection, row) {
        (_, Some(row)) => (
            row.entry.label.clone(),
            download_size(&row).map(|size| match row.state {
                ModelState::Downloading(_) => format!("downloading, {size}"),
                ModelState::NotDownloaded
                | ModelState::Downloaded
                | ModelState::Verifying
                | ModelState::Failed(_)
                | ModelState::NeedsRepair(_)
                | ModelState::Unavailable { .. } => format!("{size} download"),
            }),
            row.entry.id,
        ),
        (None, None) => (saved.clone(), Some("loading…".to_owned()), saved),
        (Some(_), None) => (saved.clone(), Some("not available".to_owned()), saved),
    };
    rsx! {
        Pill {
            id: id.clone(),
            icon: IconName::Cube,
            label: "Model",
            value,
            detail,
            hint: None::<String>,
            expanded: open(),
            disabled,
            onclick: move |_| open.toggle(),
        }
        if open() {
            ModelPicker {
                anchor: id,
                selected,
                on_close: move |()| open.set(false),
                on_pick,
            }
        }
    }
}
