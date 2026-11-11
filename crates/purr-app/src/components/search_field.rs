use super::icon::{Icon, IconName, IconSize};
use crate::utils::dom_id::use_dom_id;
use dioxus::prelude::*;

/// A rounded search box with a magnifier and a clear button; `label` names it for screen readers.
#[component]
pub fn SearchField(
    value: String,
    oninput: EventHandler<String>,
    #[props(into)] placeholder: String,
    #[props(into)] label: String,
    #[props(into)] id: Option<String>,
    #[props(into)] described_by: Option<String>,
    onkeydown: Option<EventHandler<KeyboardEvent>>,
    /// The id of the list the field filters, for a combobox.
    #[props(into)]
    controls: Option<String>,
    /// The id of the list row the keyboard points at.
    #[props(into)]
    active_descendant: Option<String>,
    #[props(into, default)] style: String,
) -> Element {
    let generated = use_dom_id("search");
    let id = id.unwrap_or(generated);
    let has_text = !value.is_empty();
    rsx! {
        label { class: "field", r#for: id.clone(), style,
            Icon { name: IconName::Search, size: IconSize::Small }
            input {
                id,
                r#type: "search",
                value,
                placeholder,
                aria_label: label,
                aria_describedby: described_by,
                aria_controls: controls,
                aria_activedescendant: active_descendant,
                autocomplete: "off",
                spellcheck: "false",
                oninput: move |event| oninput.call(event.value()),
                onkeydown: move |event: KeyboardEvent| {
                    if let Some(handler) = onkeydown {
                        handler.call(event);
                    }
                },
            }
            if has_text {
                button {
                    class: "clear",
                    r#type: "button",
                    aria_label: "Clear search",
                    onclick: move |_| oninput.call(String::new()),
                    Icon { name: IconName::Close, size: IconSize::Small }
                }
            }
        }
    }
}
