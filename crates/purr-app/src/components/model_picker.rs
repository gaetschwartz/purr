use super::{
    icon::{Icon, IconName, IconSize},
    list_picker::{LAST_USED_KEY, LastUsedOption, ListPicker, NavRow, PickerEmpty, use_list_nav},
    model_pick_row::ModelPickRow,
    pick_target::PickTarget,
    pill::Pill,
    popover::PopoverOption,
    presence::Presence,
    segmented::{Segment, Segmented},
    spinner::Spinner,
};
use crate::utils::primary_key::use_primary_key_label;
use crate::{
    components::toast::use_toasts,
    hooks::use_models::{GpuSupport, use_models},
    main_app::Route,
    model_catalog::{
        entry::{ModelEntry, Precision},
        format::download_size,
        listing::ModelRow,
        picker::{PickerModel, PickerScope, picker_list, version_tag},
        state::ModelState,
    },
    platform::IS_WEB,
    settings_store::use_settings,
};
use dioxus::prelude::*;
use purr_common::accel::AccelPolicy;

/// Every model of `use_models()` in a [`ListPicker`], anchored on the element `anchor`.
///
/// Render it inside a [`Presence`] that follows whether it is open. Each model is one line whatever versions of it exist; its versions
/// (full precision and quantized) sit inside the line as small choices, each marked when it is on
/// this device. A scope control switches between the downloaded models, where the picker opens
/// once anything is stored, and all of them; models a newer family replaces wait behind a fold.
/// Picking a model that can run here saves it with `set_model` and calls `on_pick`, whether or not
/// it is downloaded; a run downloads it first.
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
    let nav = use_list_nav();
    let mut search = use_signal(String::new);
    let mut scope_choice = use_signal(|| None::<PickerScope>);
    let mut older_toggled = use_signal(|| false);
    let catalog = use_memo(move || models.rows());
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

    let pick_last_used = move || match settings.set_default_model(None) {
        Ok(()) => on_close.call(()),
        Err(error) => {
            toasts.error("That model can't be used", error.to_string());
        }
    };
    let last_used = target == PickTarget::Default;

    let query = search();
    let searching = !query.trim().is_empty();
    let loaded = catalog.read();
    let rows: &[ModelRow] = match loaded.as_ref() {
        Some(Ok(rows)) => rows,
        Some(Err(_)) | None => &[],
    };
    let scope = scope_choice().unwrap_or_else(|| PickerScope::initial(rows));
    let list = picker_list(rows, scope, &query, &selected);
    let older_open = older_toggled() || list.older_opens(&query, &selected);
    let older_count = list.older.len();
    let hidden = list.hidden;
    let listed = !list.is_empty();
    let mut show_all = move |()| {
        nav.clear();
        scope_choice.set(Some(PickerScope::All));
    };

    let status = match loaded.as_ref() {
        None => Some(rsx! {
            div { class: "pempty row", role: "status",
                Spinner {}
                span { "Loading models…" }
            }
        }),
        Some(Err(error)) => Some(rsx! {
            div { class: "pempty", role: "alert",
                p { "Couldn't list the models: {error}" }
            }
        }),
        Some(Ok(_)) if listed => None,
        Some(Ok(_)) => Some(match (scope, searching) {
            (PickerScope::Downloaded, true) if hidden > 0 => rsx! {
                PickerEmpty {
                    text: format!("No downloaded model matches “{}”.", query.trim()),
                    action: "Search all models",
                    on_action: show_all,
                }
            },
            (PickerScope::Downloaded | PickerScope::All, true) => rsx! {
                PickerEmpty {
                    text: format!("No model matches “{}”.", query.trim()),
                    action: "Clear search",
                    on_action: move |()| search.set(String::new()),
                }
            },
            (PickerScope::Downloaded, false) => rsx! {
                PickerEmpty {
                    text: "No models downloaded yet.",
                    action: "Show all models",
                    on_action: show_all,
                }
            },
            (PickerScope::All, false) => rsx! {
                div { class: "pempty", role: "status",
                    p { "No models are available here." }
                }
            },
        }),
    };
    drop(loaded);

    let shown: Vec<PickerModel> = list
        .models
        .iter()
        .chain(list.older.iter().filter(|_| older_open))
        .cloned()
        .collect();
    let nav_rows: Vec<NavRow> = last_used
        .then(|| NavRow::single(LAST_USED_KEY))
        .into_iter()
        .chain(shown.iter().map(|model| {
            let inner = model.versions.iter().map(|row| row.entry.id.clone());
            if model.versions.len() > 1 {
                NavRow::with_inner(model.key(), inner)
            } else {
                NavRow::single(model.key())
            }
        }))
        .collect();
    let active = nav.active(&nav_rows, searching);
    let in_model = move |model: &PickerModel, key: Option<&String>| {
        key.filter(|key| *key == model.key() || model.holds(key))
            .cloned()
    };
    let selected_in = |model: &PickerModel| model.holds(&selected).then(|| selected.clone());
    let on_enter = {
        let selected = selected.clone();
        move |key: String| {
            if key == LAST_USED_KEY {
                pick_last_used();
                return;
            }
            let lead = shown
                .iter()
                .find(|model| model.key() == key)
                .and_then(|model| model.preferred(&selected));
            let inner = || {
                shown
                    .iter()
                    .flat_map(|model| &model.versions)
                    .find(|row| row.entry.id == key)
            };
            if let Some(row) = lead.or_else(inner) {
                pick(row.clone());
            }
        }
    };

    let on_cpu = IS_WEB
        && (*models.gpu.read() == GpuSupport::Unavailable
            || settings.with(|settings| settings.performance.accel_policy())
                == AccelPolicy::CPU_ONLY);
    let segments: Vec<Segment<PickerScope>> = PickerScope::ALL
        .into_iter()
        .map(|scope| Segment::new(scope, scope.label()))
        .collect();

    rsx! {
        ListPicker {
            anchor,
            label: "Choose a model",
            list_label: "Models",
            on_close,
            wide: true,
            nav,
            rows: nav_rows,
            on_enter,
            search: "Search models",
            query,
            on_query: move |value: String| search.set(value),
            filter: rsx! {
                Segmented {
                    segments,
                    value: scope,
                    label: "Show models",
                    onchange: move |scope| {
                        nav.clear();
                        scope_choice.set(Some(scope));
                    },
                }
            },
            note: on_cpu.then(|| rsx! { "Models run on the CPU here. Smaller ones are faster." }),
            status,
            footer: rsx! {
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
            },
            if last_used {
                LastUsedOption {
                    nav,
                    highlighted: active.as_deref() == Some(LAST_USED_KEY),
                    selected: selected.is_empty(),
                    subtitle: "Start with the model you picked last",
                    onselect: move |()| pick_last_used(),
                }
            }
            for model in list.models {
                ModelPickRow {
                    key: "{model.key()}",
                    selected: selected_in(&model),
                    highlighted: in_model(&model, active.as_ref()),
                    model,
                    nav,
                    hint: default_hint.clone(),
                    onselect: pick,
                    on_alt: alt,
                }
            }
            if scope == PickerScope::Downloaded && hidden > 0 && listed {
                button {
                    class: "ph pfold",
                    r#type: "button",
                    onclick: move |_| show_all(()),
                    Icon { name: IconName::ChevronRight, size: IconSize::Small }
                    if searching {
                        "{hidden} more in all models"
                    } else {
                        "All models"
                    }
                }
            }
            if older_count > 0 {
                div { role: "group", aria_label: "Older models",
                    button {
                        class: "ph pfold",
                        r#type: "button",
                        aria_expanded: older_open,
                        onclick: move |_| older_toggled.set(!older_open),
                        Icon {
                            name: if older_open { IconName::ChevronDown } else { IconName::ChevronRight },
                            size: IconSize::Small,
                        }
                        "Older models ({older_count})"
                    }
                    if older_open {
                        for model in list.older {
                            ModelPickRow {
                                key: "{model.key()}",
                                selected: selected_in(&model),
                                highlighted: in_model(&model, active.as_ref()),
                                model,
                                nav,
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
}

/// What the pill says beside the model's name: the version when it is not full precision, and the
/// download still needed.
fn pill_detail(row: &ModelRow) -> Option<String> {
    let version = match &row.entry.precision {
        Precision::Full | Precision::Mixed => None,
        precision @ (Precision::Quantized { .. } | Precision::Other(_)) => {
            Some(version_tag(precision))
        }
    };
    let size = download_size(row).map(|size| match row.state {
        ModelState::Downloading(_) => format!("downloading, {size}"),
        ModelState::NotDownloaded
        | ModelState::Downloaded
        | ModelState::Verifying(_)
        | ModelState::Failed(_)
        | ModelState::NeedsRepair(_)
        | ModelState::Unavailable { .. } => format!("{size} download"),
    });
    match (version, size) {
        (Some(version), Some(size)) => Some(format!("{version} · {size}")),
        (version, size) => version.or(size),
    }
}

/// The run-bar pill showing the model new transcriptions use; it opens a [`ModelPicker`].
///
/// The value is the model's name; its version and any download still needed follow the caption.
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
    let (value, detail, selected) = match saved {
        None => ("Last used".to_owned(), None, String::new()),
        Some(saved) => {
            let selection = models.selection(&saved);
            let row = selection
                .as_ref()
                .and_then(|selection| selection.row().cloned());
            match (&selection, row) {
                (_, Some(row)) => (
                    if row.entry.speaker_turns {
                        format!("{} · Speaker turns", row.entry.name)
                    } else {
                        row.entry.name.clone()
                    },
                    pill_detail(&row),
                    row.entry.id,
                ),
                (None, None) => (saved.clone(), Some("loading…".to_owned()), saved),
                (Some(_), None) => (saved.clone(), Some("not available".to_owned()), saved),
            }
        }
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
        Presence { present: open(),
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
