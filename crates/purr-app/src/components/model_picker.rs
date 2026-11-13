use super::{
    icon::{Icon, IconName, IconSize},
    list_picker::{
        LAST_USED_KEY, LastUsedOption, ListPicker, PickerEmpty, PickerSection, use_list_nav,
    },
    model_choice::{ClusterChoice, ModelChoice, download_size},
    pick_target::PickTarget,
    pill::Pill,
    popover::PopoverOption,
    spinner::Spinner,
};
use crate::utils::primary_key::use_primary_key_label;
use crate::{
    components::toast::use_toasts,
    hooks::use_models::{GpuSupport, use_models},
    main_app::Route,
    model_catalog::{
        entry::ModelEntry, listing::ModelRow, picker::picker_sections, state::ModelState,
    },
    platform::IS_WEB,
    settings_store::use_settings,
};
use dioxus::prelude::*;
use purr_common::accel::AccelPolicy;

/// Every model of `use_models()` in a [`ListPicker`], anchored on the element `anchor`.
///
/// Render it only while open. Models on this device come first, each version on its own row;
/// the rest follow with their download size, quantized versions as small choices under their
/// model, and the models a newer family replaces behind a fold. Picking a model that can run
/// here saves it with `set_model` and calls `on_pick`, whether or not it is downloaded; a run
/// downloads it first.
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
    let mut older_toggled = use_signal(|| false);
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
    let (status, sections) = match models.rows() {
        None => (
            Some(rsx! {
                div { class: "pempty row", role: "status",
                    Spinner {}
                    span { "Loading models…" }
                }
            }),
            None,
        ),
        Some(Err(error)) => (
            Some(rsx! {
                div { class: "pempty", role: "alert",
                    p { "Couldn't list the models: {error}" }
                }
            }),
            None,
        ),
        Some(Ok(rows)) => (None, Some(picker_sections(&rows, &query))),
    };
    let sections = sections.unwrap_or_default();
    let older_open = older_toggled() || sections.older_opens(&query, &selected);
    let choices: Vec<ModelRow> = sections.choices(older_open).into_iter().cloned().collect();
    let status = status.or_else(|| {
        (sections.is_empty()).then(|| {
            if searching {
                rsx! {
                    PickerEmpty {
                        text: format!("No model matches “{}”.", query.trim()),
                        on_clear: move |()| search.set(String::new()),
                    }
                }
            } else {
                rsx! {
                    div { class: "pempty", role: "status",
                        p { "No models are available here." }
                    }
                }
            }
        })
    });

    let keys: Vec<String> = last_used
        .then(|| LAST_USED_KEY.to_owned())
        .into_iter()
        .chain(choices.iter().map(|row| row.entry.id.clone()))
        .collect();
    let active = nav.active(&keys, searching);
    let is_active = |key: &str| active.as_deref() == Some(key);
    let on_enter = move |key: String| {
        if key == LAST_USED_KEY {
            pick_last_used();
        } else if let Some(row) = choices.iter().find(|row| row.entry.id == key) {
            pick(row.clone());
        }
    };

    let on_cpu = IS_WEB
        && (*models.gpu.read() == GpuSupport::Unavailable
            || settings.with(|settings| settings.performance.accel_policy())
                == AccelPolicy::CPU_ONLY);
    let older_count = sections.older.len();

    rsx! {
        ListPicker {
            anchor,
            label: "Choose a model",
            list_label: "Models",
            on_close,
            wide: true,
            nav,
            keys,
            on_enter,
            search: "Search models",
            query,
            on_query: move |value: String| search.set(value),
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
                    highlighted: is_active(LAST_USED_KEY),
                    selected: selected.is_empty(),
                    subtitle: "Start with the model you picked last",
                    onselect: move |()| pick_last_used(),
                }
            }
            if !sections.on_device.is_empty() {
                PickerSection { title: "On this device",
                    for row in sections.on_device {
                        ModelChoice {
                            key: "{row.entry.id}",
                            selected: row.entry.id == selected,
                            highlighted: is_active(&row.entry.id),
                            row,
                            nav,
                            hint: default_hint.clone(),
                            onselect: pick,
                            on_alt: alt,
                        }
                    }
                }
            }
            if !sections.download.is_empty() {
                PickerSection { title: "Download",
                    for cluster in sections.download {
                        ClusterChoice {
                            key: "{cluster.primary.entry.id}",
                            cluster,
                            nav,
                            selected: selected.clone(),
                            active: active.clone(),
                            hint: default_hint.clone(),
                            onselect: pick,
                            on_alt: alt,
                        }
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
                        for cluster in sections.older {
                            ClusterChoice {
                                key: "{cluster.primary.entry.id}",
                                cluster,
                                nav,
                                selected: selected.clone(),
                                active: active.clone(),
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

// The run-bar pill showing the model new transcriptions use; it opens a [`ModelPicker`].
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
