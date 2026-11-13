use super::icon::{Icon, IconName, IconSize};
use dioxus::prelude::*;

/// A centred placeholder on a card: large icon well, title, optional explanation and actions.
///
/// `inline` drops the card and the icon well so it can sit inside a card that already has a frame.
#[component]
pub fn EmptyState(
    #[props(default = IconName::Cat)] icon: IconName,
    #[props(into)] title: String,
    #[props(default)] inline: bool,
    actions: Option<Element>,
    children: Element,
) -> Element {
    let has_body = children != VNode::empty();
    let class = if inline {
        "empty compact"
    } else {
        "card surface empty"
    };
    rsx! {
        section { class,
            if !inline {
                span { class: "art",
                    Icon { name: icon, size: IconSize::ExtraLarge }
                }
            }
            h2 { "{title}" }
            if has_body {
                p { {children} }
            }
            if let Some(actions) = actions {
                div { class: "acts", {actions} }
            }
        }
    }
}
