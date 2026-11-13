use super::{
    icon::{Icon, IconName, IconSize},
    pick_target::PickTarget,
    pill::Pill,
    popover::{OptionRole, Popover, PopoverOption},
    search_field::SearchField,
    segmented::{Segment, Segmented},
    spinner::Spinner,
};
use crate::{
    components::toast::use_toasts,
    hooks::use_models::{GpuSupport, use_models},
    main_app::Route,
    model_catalog::{
        copy,
        entry::ModelEntry,
        family::ModelGroup,
        format::{check_percent, percent},
        listing::ModelRow,
        picker_filter::{PickerPrecision, PickerQuery, PickerStorage, opening_storage, pick_rows},
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
        ModelState::Verifying(progress) => check_percent(progress).map_or_else(
            || "Checking files…".to_owned(),
            |done| format!("Checking files… {done}%"),
        ),
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
        | ModelState::Verifying(_)
        | ModelState::Failed(_)
        | ModelState::NeedsRepair(_)
        | ModelState::Unavailable {
            installed: false, ..
        } => Some(ByteSize(row.entry.size_bytes).to_string()),
    }
}

/// Rows of `group`, in the order they came in.
fn in_group(rows: &[ModelRow], group: ModelGroup) -> Vec<ModelRow> {
    rows.iter()
        .filter(|row| row.entry.group == group)
        .cloned()
        .collect()
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
    /// The id of the selected model; empty when none is, as with Last used.
    #[props(into)]
    selected: String,
    #[props(default)] target: PickTarget,
    on_close: EventHandler<()>,
    /// Called after a model was picked and saved.
    on_pick: Option<EventHandler<ModelEntry>>,
) -> Element {
    let models = use_models();
    let settings = use_settings();
    let toasts = use_toasts();
    let navigator = use_navigator();
    let mut storage = use_signal(|| None::<PickerStorage>);
    let mut precision = use_signal(PickerPrecision::default);
    let mut search = use_signal(String::new);

    let pick = move |row: ModelRow| {
        if matches!(row.state, ModelState::Unavailable { .. }) {
            return;
        }
        let saved = match target {
            PickTarget::Current => settings.set_model(row.entry.id.clone()),
            PickTarget::Default => settings.set_default_model(Some(row.entry.id.clone())),
        };
        match saved {
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
            let query = PickerQuery {
                storage: storage().unwrap_or_else(|| opening_storage(&rows)),
                precision: precision(),
                search: search(),
            };
            let shown = pick_rows(&rows, &query);
            let on_device_only = query.storage == PickerStorage::OnDevice;
            let empty = shown.is_empty();
            let groups = ModelGroup::ORDER
                .into_iter()
                .map(|group| (group, in_group(&shown, group)))
                .filter(|(_, rows)| !rows.is_empty());
            rsx! {
                SearchField {
                    value: query.search.clone(),
                    placeholder: "Search models",
                    label: "Search models",
                    oninput: move |value: String| search.set(value),
                }
                div { class: "pfilters",
                    Segmented {
                        segments: vec![
                            Segment::new(PickerStorage::OnDevice, "Downloaded"),
                            Segment::new(PickerStorage::All, "All"),
                        ],
                        value: query.storage,
                        label: "Which models to list",
                        onchange: move |value| storage.set(Some(value)),
                    }
                    Segmented {
                        segments: vec![
                            Segment::new(PickerPrecision::All, "Any"),
                            Segment::new(PickerPrecision::Full, "Full"),
                            Segment::new(PickerPrecision::Quantized, "Quantized"),
                        ],
                        value: query.precision,
                        label: "Which precisions to list",
                        onchange: move |value| precision.set(value),
                    }
                }
                if empty {
                    div { class: "pempty", role: "status",
                        if on_device_only {
                            p { "No downloaded model matches." }
                            button {
                                class: "btn plain sm",
                                r#type: "button",
                                onclick: move |_| storage.set(Some(PickerStorage::All)),
                                "Show all models"
                            }
                        } else {
                            p { "No model matches." }
                        }
                    }
                }
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
        Popover { anchor, label: "Choose a model", on_close, wide: true,
            if on_cpu {
                p { class: "foot", role: "note", style: "padding:8px 12px;margin:0",
                    "Models run on the CPU here. Smaller ones are faster."
                }
            }
            if target == PickTarget::Default {
                div { class: "plist", role: "listbox", aria_label: "Default model",
                    PopoverOption {
                        title: "Last used",
                        subtitle: "Start with the model you picked last".to_owned(),
                        role: OptionRole::Option,
                        selected: selected.is_empty(),
                        leading: rsx! {
                            if selected.is_empty() {
                                Icon { name: IconName::Check, size: IconSize::Small }
                            }
                        },
                        onselect: move |()| {
                            match settings.set_default_model(None) {
                                Ok(()) => on_close.call(()),
                                Err(error) => {
                                    toasts.error("That model can't be used", error.to_string());
                                }
                            }
                        },
                    }
                }
                hr {}
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
    let on_device = row.state.is_on_device();
    let prefix = match row.state {
        ModelState::Downloading(_) => "downloaded",
        ModelState::NotDownloaded
        | ModelState::Downloaded
        | ModelState::Verifying(_)
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
            status: on_device.then(|| rsx! {
                Icon { name: IconName::CheckCircle, size: IconSize::Small }
                span { class: "sr", "Downloaded" }
            }),
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
    #[props(default)] target: PickTarget,
    /// Called after a model was picked and saved.
    on_pick: Option<EventHandler<ModelEntry>>,
) -> Element {
    let models = use_models();
    let settings = use_settings();
    let mut open = use_signal(|| false);
    let saved = settings.with(|settings| match target {
        PickTarget::Current => Some(settings.model.model_name.clone()),
        PickTarget::Default => settings.model.default_model.clone(),
    });
    let Some(saved) = saved else {
        return rsx! {
            Pill {
                id: id.clone(),
                icon: IconName::Cube,
                label: "Model",
                value: "Last used",
                detail: None::<String>,
                hint: None::<String>,
                expanded: open(),
                disabled,
                onclick: move |_| open.toggle(),
            }
            if open() {
                ModelPicker {
                    anchor: id,
                    selected: String::new(),
                    target,
                    on_close: move |()| open.set(false),
                    on_pick,
                }
            }
        };
    };
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
                | ModelState::Verifying(_)
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
                target,
                on_close: move |()| open.set(false),
                on_pick,
            }
        }
    }
}
