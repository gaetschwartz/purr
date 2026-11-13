use super::{
    icon::{Icon, IconName, IconSize},
    list_picker::ListNav,
    spinner::Spinner,
    version_chip::VersionChip,
};
use crate::{
    model_catalog::{
        copy,
        listing::ModelRow,
        picker::{Mark, PickerModel},
        state::ModelState,
    },
    utils::byte_size::ByteSize,
};
use dioxus::prelude::*;

/// Picks `row`, or makes it the default when Ctrl or Cmd is held and `on_alt` is given.
pub fn choose(
    event: &MouseEvent,
    row: &ModelRow,
    onselect: EventHandler<ModelRow>,
    on_alt: Option<EventHandler<ModelRow>>,
) {
    if matches!(row.state, ModelState::Unavailable { .. }) {
        return;
    }
    let modifiers = event.modifiers();
    match on_alt {
        Some(alt) if modifiers.ctrl() || modifiers.meta() => alt.call(row.clone()),
        Some(_) | None => onselect.call(row.clone()),
    }
}

/// A tooltip line, then the Ctrl+Click hint when there is one.
pub fn tooltip(text: String, hint: Option<&String>) -> String {
    hint.map_or_else(|| text.clone(), |hint| format!("{text}\n{hint}"))
}

fn status_view(status: Option<Mark>) -> Element {
    match status {
        None | Some(Mark::Stored) => rsx! {
            span { class: "ml-mk" }
        },
        Some(Mark::Download { bytes }) => rsx! {
            span { class: "ml-mk",
                Icon { name: IconName::Download, size: IconSize::Small }
                "{ByteSize(bytes)}"
            }
        },
        Some(Mark::Downloading { percent }) => rsx! {
            span { class: "ml-mk busy",
                Spinner {}
                if let Some(done) = percent {
                    "{done}%"
                }
            }
        },
        Some(Mark::Checking) => rsx! {
            span { class: "ml-mk busy",
                Spinner {}
                "Checking"
            }
        },
        Some(Mark::Failed) => rsx! {
            span { class: "ml-mk warn",
                Icon { name: IconName::Warning, size: IconSize::Small }
                "Failed"
            }
        },
        Some(Mark::Damaged) => rsx! {
            span { class: "ml-mk warn",
                Icon { name: IconName::Warning, size: IconSize::Small }
                "Damaged"
            }
        },
        Some(Mark::Unavailable(reason)) => rsx! {
            span { class: "ml-mk", "{copy::unavailable(reason).0}" }
        },
    }
}

/// One model of the model picker, listed once: a check when one of its versions is selected, the
/// name, its versions under the list's shared `columns`, and what picking it would still need.
///
/// Clicking the line picks [`PickerModel::preferred`]; clicking a version picks that version.
/// `selected` and `highlighted` are given only when they fall on this model, so moving the mark
/// elsewhere leaves the line untouched.
#[component]
pub fn ModelPickRow(
    model: PickerModel,
    /// The version tags every line lays its versions under.
    columns: Vec<String>,
    nav: ListNav,
    /// The selected model id, when it is one of this model's versions.
    selected: Option<String>,
    /// The marked key, when it is this line or one of its versions.
    highlighted: Option<String>,
    hint: Option<String>,
    onselect: EventHandler<ModelRow>,
    on_alt: Option<EventHandler<ModelRow>>,
) -> Element {
    let chosen = selected.clone().unwrap_or_default();
    let preferred = model.preferred(&chosen).cloned();
    let preferred_id = preferred.as_ref().map(|row| row.entry.id.clone());
    let status = model.status(&chosen);
    let key = model.key().to_owned();
    let is_selected = selected.is_some();
    let lit = highlighted.as_deref() == Some(key.as_str());
    let disabled = preferred.is_none();
    let class = format!(
        "mline{}{}{}",
        if is_selected { " sel" } else { "" },
        if highlighted.is_some() { " kb" } else { "" },
        if disabled { " off" } else { "" },
    );
    let title = model.title.clone();
    let state = model.mark(&chosen).text();
    let label = format!("{title}, {state}");
    let tip = tooltip(state, hint.as_ref());
    let slots: Vec<Option<ModelRow>> = model
        .slots(&columns)
        .into_iter()
        .map(|row| row.cloned())
        .collect();
    rsx! {
        div { class, role: "group", aria_label: "{title}",
            button {
                class: "ml-hit",
                id: nav.row_id(&key),
                r#type: "button",
                role: "option",
                aria_label: label,
                aria_selected: false,
                aria_disabled: disabled.then_some("true"),
                title: tip,
                onmousemove: {
                    let key = key.clone();
                    move |_| nav.point(&key)
                },
                onclick: move |event: MouseEvent| {
                    if let Some(row) = &preferred {
                        choose(&event, row, onselect, on_alt);
                    }
                },
            }
            span { class: "ml-ck", aria_hidden: "true",
                if is_selected {
                    Icon { name: IconName::Check, size: IconSize::Small }
                }
            }
            span { class: "ml-nm", aria_hidden: "true", "{title}" }
            span { class: "ml-vers",
                for (index , slot) in slots.into_iter().enumerate() {
                    if let Some(row) = slot {
                        VersionChip {
                            key: "{row.entry.id}",
                            selected: selected.as_deref() == Some(row.entry.id.as_str()),
                            highlighted: highlighted.as_deref() == Some(row.entry.id.as_str())
                                || (lit && preferred_id.as_deref() == Some(row.entry.id.as_str())),
                            model: title.clone(),
                            row,
                            nav,
                            hint: hint.clone(),
                            onselect,
                            on_alt,
                        }
                    } else {
                        span { key: "gap-{index}", class: "ml-ver gap" }
                    }
                }
            }
            {status_view(status)}
        }
    }
}
