use super::{
    icon::{Icon, IconName, IconSize},
    list_picker::{LAST_USED_KEY, LastUsedOption, ListPicker, NavRow, PickerEmpty, use_list_nav},
    pick_target::PickTarget,
    pill::Pill,
    popover::{OptionRole, PopoverOption},
    presence::Presence,
};
use crate::utils::primary_key::use_primary_key_label;
use crate::{
    components::toast::use_toasts,
    hooks::use_models::use_models,
    settings_store::use_settings,
    transcribe::language_choice::{choices, for_model, is_locked, note},
};
use dioxus::prelude::*;
use purr_common::{
    language::{choice::LanguageChoice, known},
    model::profile::ModelProfile,
};

const AUTOMATIC_HINT: &str = "Detects the language from the first 30 seconds";

/// The run-bar pill showing the spoken language; it opens a [`LanguagePicker`].
///
/// With an English-only model selected it shows English, whatever the setting says.
#[component]
pub fn LanguagePill(
    /// The pill's DOM id, which the popover hangs from.
    #[props(into)]
    id: String,
    #[props(default)] disabled: bool,
    #[props(default)] target: PickTarget,
) -> Element {
    let settings = use_settings();
    let models = use_models();
    let mut open = use_signal(|| false);
    let (saved, model) = settings.with(|settings| {
        (
            match target {
                PickTarget::Current => Some(settings.ui.language.clone()),
                PickTarget::Default => settings.ui.default_language.clone(),
            },
            settings.model.model_name.clone(),
        )
    });
    let profile = match target {
        PickTarget::Current => models.find(&model).map(|info| info.descriptor.profile),
        PickTarget::Default => None,
    };
    let locked = profile.as_ref().is_some_and(is_locked);
    let (value, detail, hint, selected, profile) = match saved {
        None => ("Last used".to_owned(), None, None, String::new(), None),
        Some(selected) => {
            let choice = for_model(LanguageChoice::from_setting(&selected), profile.as_ref());
            let (detail, hint) = match (&choice, locked) {
                (LanguageChoice::Spoken(_), true) => {
                    (Some(format!("{} only", choice.label())), None)
                }
                (LanguageChoice::Automatic, _) => {
                    (None, Some(format!("Automatic · {AUTOMATIC_HINT}")))
                }
                (LanguageChoice::Spoken(_), false) => (None, None),
            };
            (choice.label(), detail, hint, selected, profile)
        }
    };
    rsx! {
        Pill {
            id: id.clone(),
            icon: IconName::Globe,
            label: "Spoken language",
            value,
            detail,
            hint,
            expanded: open(),
            disabled,
            onclick: move |_| open.toggle(),
        }
        Presence { present: open(),
            LanguagePicker {
                anchor: id,
                selected,
                profile,
                target,
                on_close: move |()| open.set(false),
            }
        }
    }
}

/// The spoken languages in a [`ListPicker`] with a search field, anchored on `anchor`.
///
/// Render it inside a [`Presence`] that follows whether it is open. "Automatic" comes first; arrows move through the list from the
/// search field and Enter picks. Picking saves the setting with `set_language`. With
/// a `profile` that leaves nothing to choose the list holds what the model takes.
#[component]
pub fn LanguagePicker(
    /// The id of the element the popover hangs from.
    #[props(into)]
    anchor: String,
    /// The language setting: a Whisper code or `auto`; empty when none is, as with Last used.
    #[props(into)]
    selected: String,
    /// What the chosen model does with a language; absent when no model is known.
    #[props(default)]
    profile: Option<ModelProfile>,
    #[props(default)] target: PickTarget,
    on_close: EventHandler<()>,
) -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let nav = use_list_nav();
    let mut query = use_signal(String::new);

    let restricted = profile.as_ref().is_some_and(is_locked);
    let none_selected = selected.is_empty();
    let current = (!none_selected)
        .then(|| for_model(LanguageChoice::from_setting(&selected), profile.as_ref()));
    let choices = choices(profile.as_ref(), &query());
    let note = profile.as_ref().and_then(note);
    let last_used = target == PickTarget::Default;
    let rows: Vec<NavRow> = last_used
        .then(|| NavRow::single(LAST_USED_KEY))
        .into_iter()
        .chain(
            choices
                .iter()
                .map(|choice| NavRow::single(choice.setting())),
        )
        .collect();
    let active = nav.active(&rows, !query().trim().is_empty());
    let is_active = |key: &str| active.as_deref() == Some(key);

    let choose = move |choice: LanguageChoice| {
        if restricted {
            on_close.call(());
            return;
        }
        let saved = match target {
            PickTarget::Current => settings.set_language(choice.setting()),
            PickTarget::Default => settings.set_default_language(Some(choice.setting())),
        };
        match saved {
            Ok(()) => on_close.call(()),
            Err(error) => {
                toasts.error("That language can't be used", error.to_string());
            }
        }
    };
    let choose_last_used = move || match settings.set_default_language(None) {
        Ok(()) => on_close.call(()),
        Err(error) => {
            toasts.error("That language can't be used", error.to_string());
        }
    };

    let primary_key = use_primary_key_label();
    let alt_enabled = target == PickTarget::Current && !restricted;
    let default_hint = alt_enabled.then(|| format!("{}+Click to set as default", primary_key()));
    let choose_default = move |choice: LanguageChoice| {
        let code = choice.setting();
        let saved = settings.update(|settings| {
            settings.ui.language.clone_from(&code);
            settings.ui.default_language = Some(code);
        });
        match saved {
            Ok(()) => {
                toasts.success("Default language set", choice.label());
                on_close.call(());
            }
            Err(error) => {
                toasts.error("That language can't be used", error.to_string());
            }
        }
    };

    let enter_choices = choices.clone();
    let on_enter = move |key: String| {
        if key == LAST_USED_KEY {
            choose_last_used();
        } else if let Some(choice) = enter_choices.iter().find(|choice| choice.setting() == key) {
            choose(choice.clone());
        }
    };

    let status = choices.is_empty().then(|| {
        rsx! {
            PickerEmpty {
                text: format!("No language called “{}”.", query().trim()),
                action: "Clear search",
                on_action: move |()| query.set(String::new()),
            }
        }
    });

    rsx! {
        ListPicker {
            anchor,
            label: "Spoken language",
            list_label: "Languages",
            on_close,
            nav,
            rows,
            on_enter,
            search: (!restricted).then(|| format!("Search {} languages", known::all().len())),
            query: query(),
            on_query: move |value: String| query.set(value),
            note: note.map(|text| rsx! { "{text}" }),
            status,
            if last_used {
                LastUsedOption {
                    nav,
                    highlighted: is_active(LAST_USED_KEY),
                    selected: none_selected,
                    subtitle: "Start with the language you picked last",
                    onselect: move |()| choose_last_used(),
                }
            }
            for choice in choices {
                PopoverOption {
                    key: "{choice.setting()}",
                    id: nav.row_id(&choice.setting()),
                    title: choice.label(),
                    subtitle: (choice == LanguageChoice::Automatic).then(|| AUTOMATIC_HINT.to_owned()),
                    role: OptionRole::Option,
                    selected: current.as_ref() == Some(&choice),
                    highlighted: is_active(&choice.setting()),
                    hint: default_hint.clone(),
                    on_alt: alt_enabled
                        .then(|| {
                            let choice = choice.clone();
                            EventHandler::new(move |()| choose_default(choice.clone()))
                        }),
                    on_hover: {
                        let key = choice.setting();
                        move |()| nav.point(&key)
                    },
                    leading: rsx! {
                        if current.as_ref() == Some(&choice) {
                            Icon { name: IconName::Check, size: IconSize::Small }
                        }
                    },
                    onselect: {
                        let choice = choice.clone();
                        move |()| choose(choice.clone())
                    },
                }
            }
        }
    }
}
