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

fn mark_view(mark: Mark) -> Element {
    match mark {
        Mark::Stored => rsx! {
            span { class: "mk ok",
                Icon { name: IconName::CheckCircle, size: IconSize::Small }
            }
        },
        Mark::Download { bytes } => rsx! {
            span { class: "mk",
                Icon { name: IconName::Download, size: IconSize::Small }
                "{ByteSize(bytes)}"
            }
        },
        Mark::Downloading { percent } => rsx! {
            span { class: "mk busy",
                Spinner {}
                if let Some(done) = percent {
                    "{done}%"
                }
            }
        },
        Mark::Checking => rsx! {
            span { class: "mk busy",
                Spinner {}
            }
        },
        Mark::Failed | Mark::Damaged => rsx! {
            span { class: "mk warn",
                Icon { name: IconName::Warning, size: IconSize::Small }
            }
        },
        Mark::Unavailable(reason) => rsx! {
            span { class: "mk", "{copy::unavailable(reason).0}" }
        },
    }
}

/// One model of the model picker, listed once: a check when one of its versions is selected, the
/// name, its versions when there are several, and a mark saying whether it is on this device.
///
/// Clicking the line picks [`PickerModel::preferred`]; clicking a version picks that version.
/// `selected` and `highlighted` are given only when they fall on this model, so moving the mark
/// elsewhere leaves the line untouched.
#[component]
pub fn ModelPickRow(
    model: PickerModel,
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
    let mark = model.mark(&chosen);
    let key = model.key().to_owned();
    let is_selected = selected.is_some();
    let lit = highlighted.as_deref() == Some(key.as_str());
    let disabled = preferred.is_none();
    let class = format!(
        "mrow{}{}{}",
        if is_selected { " sel" } else { "" },
        if lit { " kb" } else { "" },
        if disabled { " off" } else { "" },
    );
    let title = model.title.clone();
    let several = model.versions.len() > 1;
    let state = mark.text();
    let label = format!("{title}, {state}");
    let tip = tooltip(state, hint.as_ref());
    let versions = if several {
        model.versions.clone()
    } else {
        Vec::new()
    };
    rsx! {
        div { class, role: "group", aria_label: "{title}",
            button {
                class: "mhit",
                id: nav.row_id(&key),
                r#type: "button",
                role: "option",
                aria_label: label,
                aria_selected: is_selected && !several,
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
            span { class: "ck", aria_hidden: "true",
                if is_selected {
                    Icon { name: IconName::Check, size: IconSize::Small }
                }
            }
            span { class: "nm", aria_hidden: "true", "{title}" }
            if several {
                span { class: "vers",
                    for row in versions {
                        VersionChip {
                            key: "{row.entry.id}",
                            selected: selected.as_deref() == Some(row.entry.id.as_str()),
                            highlighted: highlighted.as_deref() == Some(row.entry.id.as_str()),
                            model: title.clone(),
                            row,
                            nav,
                            hint: hint.clone(),
                            onselect,
                            on_alt,
                        }
                    }
                }
            }
            {mark_view(mark)}
        }
    }
}
