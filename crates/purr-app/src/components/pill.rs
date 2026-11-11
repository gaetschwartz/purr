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
    let accessible = detail.as_ref().map_or_else(
        || format!("{label}: {value}. Change {}", label.to_lowercase()),
        |detail| {
            format!(
                "{label}: {value}, {detail}. Change {}",
                label.to_lowercase()
            )
        },
    );
    rsx! {
        button {
            id,
            class: "pill",
            r#type: "button",
            aria_haspopup: "dialog",
            aria_expanded: expanded,
            aria_label: accessible,
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
