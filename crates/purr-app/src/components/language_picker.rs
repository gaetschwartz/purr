use super::{
    icon::{Icon, IconName, IconSize},
    pill::Pill,
    popover::{OptionRole, Popover, PopoverOption},
    search_field::SearchField,
};
use crate::{
    components::toast::use_toasts,
    hooks::use_models::use_models,
    model_catalog::entry::{Language as ModelLanguage, ModelEntry},
    settings_store::use_settings,
    transcribe::{
        language_choice::{LanguageChoice, options},
        search::{next, previous},
    },
    utils::dom_id::use_dom_id,
};
use dioxus::prelude::*;
use purr_common::Language;

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
) -> Element {
    let settings = use_settings();
    let models = use_models();
    let mut open = use_signal(|| false);
    let (selected, model) = settings.with(|settings| {
        (
            settings.ui.language.clone(),
            settings.model.model_name.clone(),
        )
    });
    let english_only = models
        .find(&model)
        .is_some_and(|info| ModelEntry::from_info(&info).language == ModelLanguage::EnglishOnly);
    let choice = LanguageChoice::from_setting(&selected).for_model(english_only);
    let detail = match (&choice, english_only) {
        (_, true) => Some("model is English only"),
        (LanguageChoice::Automatic, false) => Some("detect"),
        (LanguageChoice::Spoken(_), false) => None,
    };
    rsx! {
        Pill {
            id: id.clone(),
            icon: IconName::Globe,
            label: "Spoken language",
            value: choice.label(),
            detail: detail.map(str::to_owned),
            expanded: open(),
            disabled,
            onclick: move |_| open.toggle(),
        }
        if open() {
            LanguagePicker {
                anchor: id,
                selected,
                english_only,
                on_close: move |()| open.set(false),
            }
        }
    }
}

/// Whisper's languages in a popover (a sheet on phones) with a search field, anchored on `anchor`.
///
/// Render it only while open. "Automatic" comes first; arrows move through the list from the
/// search field and Enter picks. Picking saves the setting with `set_language`. With
/// `english_only` the list is locked to English, because the model knows no other language.
#[component]
pub fn LanguagePicker(
    /// The id of the element the popover hangs from.
    #[props(into)]
    anchor: String,
    /// The `ui.language` setting: a Whisper code or `auto`.
    #[props(into)]
    selected: String,
    /// The chosen model only transcribes English.
    #[props(default)]
    english_only: bool,
    on_close: EventHandler<()>,
) -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let list_id = use_dom_id("languages");
    let mut query = use_signal(String::new);
    let mut highlight = use_signal(|| 0_usize);

    let current = LanguageChoice::from_setting(&selected).for_model(english_only);
    let choices = if english_only {
        vec![LanguageChoice::Spoken(Language::En)]
    } else {
        options(&query())
    };
    let count = choices.len();
    let active = highlight().min(count.saturating_sub(1));
    let row_id = {
        let list_id = list_id.clone();
        move |index: usize| format!("{list_id}-{index}")
    };
    let active_id = (count > 0).then(|| row_id(active));

    let choose = move |choice: LanguageChoice| {
        if english_only {
            on_close.call(());
            return;
        }
        match settings.set_language(choice.setting()) {
            Ok(()) => on_close.call(()),
            Err(error) => {
                toasts.error("That language can't be used", error.to_string());
            }
        }
    };

    let keyboard_choices = choices.clone();
    let reveal_id = row_id.clone();
    let on_key = move |event: KeyboardEvent| {
        let moved = match event.key() {
            Key::Named(NamedKey::ArrowDown) => Some(next(active, count)),
            Key::Named(NamedKey::ArrowUp) => Some(previous(active, count)),
            Key::Named(NamedKey::Enter) => {
                event.prevent_default();
                if let Some(choice) = keyboard_choices.get(active) {
                    choose(choice.clone());
                }
                None
            }
            Key::Named(_) | Key::Character(_) => None,
        };
        if let Some(index) = moved {
            event.prevent_default();
            highlight.set(index);
            let script = format!(
                "document.getElementById({:?})?.scrollIntoView({{block:'nearest'}})",
                reveal_id(index)
            );
            document::eval(&script);
        }
    };

    rsx! {
        Popover { anchor, label: "Spoken language", on_close,
            if english_only {
                p { class: "foot", style: "margin:6px 10px",
                    "This model only transcribes English."
                }
            } else {
                SearchField {
                    value: query(),
                    placeholder: format!("Search {} languages", Language::VARIANTS.len() - 1),
                    label: "Search languages",
                    controls: list_id.clone(),
                    active_descendant: active_id,
                    oninput: move |value: String| {
                        query.set(value);
                        highlight.set(0);
                    },
                    onkeydown: on_key,
                }
            }
            div { class: "plist", role: "listbox", id: list_id, aria_label: "Languages",
                if choices.is_empty() {
                    p { class: "faint", style: "padding:10px;margin:0;font-size:14px",
                        "No language called “{query}”."
                    }
                }
                for (index , choice) in choices.into_iter().enumerate() {
                    PopoverOption {
                        key: "{choice.setting()}",
                        id: row_id(index),
                        title: choice.label(),
                        subtitle: (choice == LanguageChoice::Automatic).then(|| AUTOMATIC_HINT.to_owned()),
                        role: OptionRole::Option,
                        selected: choice == current,
                        highlighted: index == active,
                        leading: rsx! {
                            if choice == current {
                                Icon { name: IconName::Check, size: IconSize::Small }
                            }
                        },
                        onselect: move |()| choose(choice.clone()),
                    }
                }
            }
        }
    }
}
