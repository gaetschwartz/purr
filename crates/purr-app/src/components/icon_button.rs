use super::icon::{Icon, IconName, IconSize};
use dioxus::prelude::*;

/// A round, icon-only button; `label` becomes its accessible name and tooltip.
#[component]
pub fn IconButton(
    icon: IconName,
    #[props(into)] label: String,
    #[props(default = IconSize::Medium)] size: IconSize,
    #[props(default)] disabled: bool,
    onclick: Option<EventHandler<MouseEvent>>,
    #[props(extends = GlobalAttributes)] attributes: Vec<Attribute>,
) -> Element {
    rsx! {
        button {
            class: "icon-btn",
            r#type: "button",
            aria_label: label.clone(),
            title: label,
            disabled,
            onclick: move |event| {
                if let Some(handler) = onclick {
                    handler.call(event);
                }
            },
            ..attributes,
            Icon { name: icon, size }
        }
    }
}
