use super::{
    icon::{Icon, IconName, IconSize},
    pick_target::PickTarget,
    pill::Pill,
    popover::{OptionRole, Popover, PopoverOption},
    search_field::SearchField,
    segmented::{Segment, Segmented},
    spinner::Spinner,
};
use crate::utils::primary_key::use_primary_key_label;
use crate::{
    components::toast::use_toasts,
    hooks::use_models::{GpuSupport, use_models},
    main_app::Route,
    model_catalog::{
        clusters::{ModelCluster, cluster},
        copy,
        entry::{ModelEntry, Precision},
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

fn subtitle(row: &ModelRow) -> Option<String> {
    match &row.state {
        ModelState::Unavailable { reason, .. } => Some(copy::unavailable(*reason).1.to_owned()),
        ModelState::Downloading(_) => Some("Downloading…".to_owned()),
        ModelState::Verifying(progress) => Some(check_percent(progress).map_or_else(
            || "Checking files…".to_owned(),
            |done| format!("Checking files… {done}%"),
        )),
        ModelState::Failed(_) => Some("Last download failed".to_owned()),
        ModelState::NeedsRepair(_) => Some("Files damaged".to_owned()),
        ModelState::NotDownloaded | ModelState::Downloaded => None,
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
    let mut more_open = use_signal(|| false);
    let primary_key = use_primary_key_label();
    let default_hint = (target == PickTarget::Current)
        .then(|| format!("{}+Click to set as default", primary_key()));

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

    let pick_default = move |row: ModelRow| {
        if matches!(row.state, ModelState::Unavailable { .. }) {
            return;
        }
        let id = row.entry.id.clone();
        let saved = settings.update(|settings| {
            settings.model.model_name.clone_from(&id);
            settings.model.default_model = Some(id);
        });
        match saved {
            Ok(()) => {
                toasts.success("Default model set", row.entry.label.clone());
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
    let alt = (target == PickTarget::Current).then_some(EventHandler::new(pick_default));

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
            let (older, current): (Vec<ModelCluster>, Vec<ModelCluster>) = cluster(&shown)
                .into_iter()
                .partition(ModelCluster::superseded);
            let older_count = older.len();
            let older_holds_selected = older
                .iter()
                .any(|cluster| cluster.rows().any(|row| row.entry.id == selected));
            let older_open = more_open() || !query.search.trim().is_empty() || older_holds_selected;
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
                div { class: "plist", role: "listbox", aria_label: "Models",
                    for cluster in current {
                        ClusterOption {
                            key: "{cluster.primary.entry.id}",
                            cluster: cluster.clone(),
                            selected: selected.clone(),
                            hint: default_hint.clone(),
                            onselect: pick,
                            on_alt: alt,
                        }
                    }
                    if older_count > 0 {
                        button {
                            class: "btn plain sm pmore",
                            r#type: "button",
                            aria_expanded: older_open,
                            onclick: move |_| more_open.toggle(),
                            "More models ({older_count})"
                        }
                        if older_open {
                            for cluster in older {
                                ClusterOption {
                                    key: "{cluster.primary.entry.id}",
                                    cluster: cluster.clone(),
                                    selected: selected.clone(),
                                    hint: default_hint.clone(),
                                    onselect: pick,
                                    on_alt: alt,
                                }
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
fn ClusterOption(
    cluster: ModelCluster,
    selected: String,
    hint: Option<String>,
    onselect: EventHandler<ModelRow>,
    on_alt: Option<EventHandler<ModelRow>>,
) -> Element {
    let name = cluster.primary.entry.name.clone();
    rsx! {
        ModelOption {
            row: cluster.primary.clone(),
            selected: cluster.primary.entry.id == selected,
            hint: hint.clone(),
            onselect,
            on_alt,
        }
        if !cluster.variants.is_empty() {
            div {
                class: "pvars",
                role: "group",
                aria_label: "Other versions of {name}",
                for row in cluster.variants.clone() {
                    VariantOption {
                        key: "{row.entry.id}",
                        selected: row.entry.id == selected,
                        row,
                        hint: hint.clone(),
                        onselect,
                        on_alt,
                    }
                }
            }
        }
    }
}

#[component]
fn VariantOption(
    row: ModelRow,
    selected: bool,
    hint: Option<String>,
    onselect: EventHandler<ModelRow>,
    on_alt: Option<EventHandler<ModelRow>>,
) -> Element {
    let disabled = matches!(row.state, ModelState::Unavailable { .. });
    let tag = match &row.entry.precision {
        Precision::Quantized { bits } => format!("Q{bits}"),
        Precision::Other(raw) => raw.clone(),
        Precision::Full | Precision::Mixed => "Full".to_owned(),
    };
    let size = download_size(&row);
    let on_device = row.state.is_on_device();
    rsx! {
        button {
            class: if selected { "pvar sel" } else { "pvar" },
            r#type: "button",
            role: "option",
            title: hint,
            aria_selected: selected,
            aria_disabled: disabled.then_some("true"),
            onclick: move |event: MouseEvent| {
                if disabled {
                    return;
                }
                let modifiers = event.modifiers();
                match on_alt {
                    Some(alt) if modifiers.ctrl() || modifiers.meta() => alt.call(row.clone()),
                    Some(_) | None => onselect.call(row.clone()),
                }
            },
            b { "{tag}" }
            if on_device {
                Icon { name: IconName::CheckCircle, size: IconSize::Small }
                span { class: "sr", "Downloaded" }
            } else if let Some(size) = size {
                span { "{size}" }
            }
        }
    }
}

#[component]
fn ModelOption(
    row: ModelRow,
    selected: bool,
    hint: Option<String>,
    onselect: EventHandler<ModelRow>,
    on_alt: Option<EventHandler<ModelRow>>,
) -> Element {
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
            hint,
            on_alt: on_alt.map(|alt| {
                let row = row.clone();
                EventHandler::new(move |()| alt.call(row.clone()))
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
