use dioxus::prelude::*;

/// An iOS-style on/off switch (`role="switch"`); name it with `label` or `labelled_by`.
#[component]
pub fn Switch(
    checked: bool,
    onchange: EventHandler<bool>,
    #[props(into)] label: Option<String>,
    #[props(into)] labelled_by: Option<String>,
    #[props(default)] disabled: bool,
    #[props(extends = GlobalAttributes)] attributes: Vec<Attribute>,
) -> Element {
    rsx! {
        button {
            class: "switch",
            r#type: "button",
            role: "switch",
            aria_checked: checked,
            aria_label: label,
            aria_labelledby: labelled_by,
            disabled,
            onclick: move |_| onchange.call(!checked),
            ..attributes,
        }
    }
}
