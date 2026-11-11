use super::icon::{Icon, IconName, IconSize};
use dioxus::prelude::*;

/// A picker button showing a caption and the current value; it opens a popover anchored on its `id`.
#[component]
pub fn Pill(
    #[props(into)] id: String,
    icon: IconName,
    #[props(into)] label: String,
    #[props(into)] value: String,
    #[props(into)] detail: Option<String>,
    #[props(default)] expanded: bool,
    #[props(default)] disabled: bool,
    onclick: EventHandler<MouseEvent>,
) -> Element {
    rsx! {
        button {
            id,
            class: "pill",
            r#type: "button",
            aria_haspopup: "dialog",
            aria_expanded: expanded,
            disabled,
            onclick: move |event| onclick.call(event),
            span { class: "ico",
                Icon { name: icon, size: IconSize::Small }
            }
            span { class: "grow",
                span { class: "lbl", "{label}" }
                span { class: "val",
                    "{value}"
                    if let Some(detail) = &detail {
                        span { class: "faint", " · {detail}" }
                    }
                }
            }
            Icon {
                name: IconName::ChevronDown,
                size: IconSize::Small,
                class: "chev",
            }
        }
    }
}
