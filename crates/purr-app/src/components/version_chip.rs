use super::{
    icon::{Icon, IconName, IconSize},
    list_picker::ListNav,
    model_pick_row::{choose, tooltip},
};
use crate::model_catalog::{
    listing::ModelRow,
    picker::{Mark, version_tag},
    state::ModelState,
};
use dioxus::prelude::*;

/// One version of a model in the model picker, such as `Full` or `Q5`, with a check-circle when it
/// is on this device.
#[component]
pub fn VersionChip(
    row: ModelRow,
    /// The name of the model the version belongs to.
    #[props(into)]
    model: String,
    nav: ListNav,
    selected: bool,
    highlighted: bool,
    hint: Option<String>,
    onselect: EventHandler<ModelRow>,
    on_alt: Option<EventHandler<ModelRow>>,
) -> Element {
    let mark = Mark::of(&row);
    let stored = matches!(mark, Mark::Stored);
    let disabled = matches!(row.state, ModelState::Unavailable { .. });
    let tag = version_tag(&row.entry.precision);
    let state = mark.text();
    let label = format!("{model} {tag}, {state}");
    let tip = tooltip(format!("{tag} · {state}"), hint.as_ref());
    let class = match (stored, highlighted) {
        (true, true) => "ver on kb",
        (true, false) => "ver on",
        (false, true) => "ver kb",
        (false, false) => "ver",
    };
    let id = row.entry.id.clone();
    rsx! {
        button {
            id: nav.row_id(&id),
            class,
            r#type: "button",
            role: "option",
            aria_label: label,
            aria_selected: selected,
            aria_disabled: disabled.then_some("true"),
            title: tip,
            onmousemove: move |_| nav.point(&id),
            onclick: move |event: MouseEvent| choose(&event, &row, onselect, on_alt),
            if stored {
                Icon { name: IconName::CheckCircle, size: IconSize::Small }
            }
            "{tag}"
        }
    }
}
