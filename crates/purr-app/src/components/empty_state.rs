use super::icon::{Icon, IconName, IconSize};
use dioxus::prelude::*;

/// A centred placeholder on a glass card: large icon well, title, explanation and actions.
#[component]
pub fn EmptyState(
    #[props(default = IconName::Cat)] icon: IconName,
    #[props(into)] title: String,
    actions: Option<Element>,
    children: Element,
) -> Element {
    rsx! {
        section { class: "card glass empty",
            span { class: "art",
                Icon { name: icon, size: IconSize::ExtraLarge }
            }
            h2 { "{title}" }
            p { {children} }
            if let Some(actions) = actions {
                div { class: "acts", {actions} }
            }
        }
    }
}
