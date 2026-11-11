use crate::utils::dom_id::use_dom_id;
use dioxus::prelude::*;

/// A titled list section of the Models screen: a small caps header with an optional hint, then a glass group.
#[component]
pub fn ModelSection(
    #[props(into)] title: String,
    #[props(into)] hint: Option<String>,
    children: Element,
) -> Element {
    let heading_id = use_dom_id("models-section");
    rsx! {
        section { aria_labelledby: "{heading_id}",
            div { class: "fam-h",
                h2 { id: "{heading_id}", "{title}" }
                if let Some(hint) = hint {
                    span { "{hint}" }
                }
            }
            div { class: "group glass", {children} }
        }
    }
}
