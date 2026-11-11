use super::{
    button::{Button, ButtonSize, ButtonVariant},
    icon::{Icon, IconName, IconSize},
    pill::Pill,
    popover::{OptionRole, Popover, PopoverOption},
    spinner::Spinner,
};
use crate::{
    components::toast::use_toasts,
    hooks::use_models::use_models,
    main_app::Route,
    model_catalog::{
        entry::ModelEntry,
        family::ModelGroup,
        format::percent,
        listing::ModelRow,
        state::{ModelState, UnavailableReason},
    },
    settings_store::use_settings,
    utils::byte_size::ByteSize,
};
use dioxus::prelude::*;

/// Why a model is greyed out in the picker.
const fn unavailable_text(reason: UnavailableReason) -> &'static str {
    match reason {
        UnavailableReason::NeedsShaderF16 => "Needs a GPU feature (shader-f16) this device lacks",
        UnavailableReason::NeedsWebGpu => "Needs WebGPU, which this browser doesn't offer",
    }
}

fn subtitle(row: &ModelRow) -> String {
    match &row.state {
        ModelState::Unavailable { reason, .. } => unavailable_text(*reason).to_owned(),
        ModelState::Downloading(_) => "Downloading…".to_owned(),
        ModelState::Verifying => "Checking the files…".to_owned(),
        ModelState::Failed(_) => "The last download failed. Pick it to try again.".to_owned(),
        ModelState::NeedsRepair(_) => {
            "Its files are damaged. Pick it to download them again.".to_owned()
        }
        ModelState::NotDownloaded | ModelState::Downloaded => row.entry.hint(),
    }
}

fn trailing(row: &ModelRow) -> String {
    match &row.state {
        ModelState::Downloading(progress) => {
            percent(progress).map_or_else(|| "…".to_owned(), |done| format!("{done}%"))
        }
        ModelState::NotDownloaded
        | ModelState::Downloaded
        | ModelState::Verifying
        | ModelState::Failed(_)
        | ModelState::NeedsRepair(_)
        | ModelState::Unavailable { .. } => ByteSize(row.entry.size_bytes).to_string(),
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

/// The models of `use_models()` in a popover (a sheet on phones), anchored on the element `anchor`.
///
/// Render it only while open. Models on this device come first, then the others grouped by size
/// class with a download offer. Picking a downloaded model saves it with `set_model` and calls
/// `on_pick`; picking another one asks to download it, then selects it. Disabled rows say why.
#[component]
pub fn ModelPicker(
    /// The id of the element the popover hangs from.
    #[props(into)]
    anchor: String,
    /// The id of the selected model.
    #[props(into)]
    selected: String,
    on_close: EventHandler<()>,
    /// Called after a downloaded model was picked and saved.
    on_pick: Option<EventHandler<ModelEntry>>,
) -> Element {
    let models = use_models();
    let settings = use_settings();
    let toasts = use_toasts();
    let navigator = use_navigator();
    let mut offer = use_signal(|| None::<ModelEntry>);

    let save = move |entry: &ModelEntry| match settings.set_model(entry.id.clone()) {
        Ok(()) => true,
        Err(error) => {
            toasts.error("That model can't be used", error.to_string());
            false
        }
    };

    let pick = move |row: ModelRow| {
        if matches!(row.state, ModelState::Unavailable { .. }) {
            return;
        }
        if row.state.is_on_device() {
            if save(&row.entry) {
                on_close.call(());
                if let Some(handler) = on_pick {
                    handler.call(row.entry);
                }
            }
        } else if !models.is_busy(&row.entry.id) {
            offer.set(Some(row.entry));
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
                "Could not list the models: {error}"
            }
        },
        Some(Ok(rows)) => {
            let on_device: Vec<ModelRow> = ModelGroup::ORDER
                .into_iter()
                .flat_map(|group| in_group(&rows, group))
                .filter(|row| row.state.is_on_device())
                .collect();
            let others = ModelGroup::ORDER
                .into_iter()
                .map(|group| {
                    let rows: Vec<ModelRow> = in_group(&rows, group)
                        .into_iter()
                        .filter(|row| !row.state.is_on_device())
                        .collect();
                    (group, rows)
                })
                .filter(|(_, rows)| !rows.is_empty());
            rsx! {
                if !on_device.is_empty() {
                    div { class: "ph", "On this device" }
                    div { class: "plist", role: "listbox", aria_label: "Downloaded models",
                        for row in on_device {
                            ModelOption {
                                key: "{row.entry.id}",
                                row: row.clone(),
                                selected: row.entry.id == selected,
                                onselect: pick,
                            }
                        }
                    }
                    hr {}
                }
                div { class: "ph", "Download another" }
                for (group , rows) in others {
                    div {
                        class: "plist",
                        role: "listbox",
                        aria_label: "{group.label()} models to download",
                        key: "{group.label()}",
                        div { class: "ph", style: "text-transform:none;letter-spacing:0", "{group.label()}" }
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

    rsx! {
        Popover { anchor, label: "Choose a model", on_close,
            if let Some(entry) = offer() {
                div { class: "dl-offer", role: "group", aria_label: "Download {entry.label}",
                    p { style: "margin:0 0 8px;font-size:14px",
                        b { "{entry.label}" }
                        " isn't on this device yet. Download it now ({ByteSize(entry.size_bytes)})? You can keep using Purr meanwhile."
                    }
                    div { class: "row wrap",
                        Button {
                            variant: ButtonVariant::Primary,
                            size: ButtonSize::Small,
                            icon: IconName::Download,
                            autofocus: true,
                            disabled: models.is_downloading(),
                            onclick: {
                                let entry = entry.clone();
                                move |_| {
                                if save(&entry) {
                                    models.download(&entry.id);
                                    on_close.call(());
                                }
                                }
                            },
                            "Download and use"
                        }
                        Button {
                            size: ButtonSize::Small,
                            onclick: move |_| offer.set(None),
                            "Not now"
                        }
                    }
                    if models.is_downloading() {
                        p { class: "foot", style: "margin:8px 0 0",
                            "Another model is downloading. Wait for it to finish, or cancel it in Models."
                        }
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
    let on_device = row.state.is_on_device();
    let size = trailing(&row);
    let trailing_text = if on_device {
        size
    } else {
        format!("↓ {size}")
    };
    rsx! {
        PopoverOption {
            title: row.entry.label.clone(),
            subtitle: subtitle(&row),
            trailing: trailing_text,
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

/// The run-bar pill showing the selected model and its size; it opens a [`ModelPicker`].
#[component]
pub fn ModelPill(
    /// The pill's DOM id, which the popover hangs from.
    #[props(into)]
    id: String,
    #[props(default)] disabled: bool,
    /// Called after a downloaded model was picked and saved.
    on_pick: Option<EventHandler<ModelEntry>>,
) -> Element {
    let models = use_models();
    let settings = use_settings();
    let mut open = use_signal(|| false);
    let selected = settings.with(|settings| settings.model.model_name.clone());
    let (value, detail) = match models.find(&selected) {
        Some(info) => {
            let entry = ModelEntry::from_info(&info);
            let detail = if models.state(&selected).is_on_device() {
                ByteSize(entry.size_bytes).to_string()
            } else {
                format!("{}, not downloaded", ByteSize(entry.size_bytes))
            };
            (entry.label, detail)
        }
        None if models.models.read().is_none() => (selected.clone(), "loading…".to_owned()),
        None => (selected.clone(), "not available here".to_owned()),
    };
    rsx! {
        Pill {
            id: id.clone(),
            icon: IconName::Cube,
            label: "Model",
            value,
            detail,
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
