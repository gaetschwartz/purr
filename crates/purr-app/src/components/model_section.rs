use crate::utils::dom_id::use_dom_id;
use dioxus::prelude::*;

/// A titled list section of the Models screen: a small caps header, then a grouped list.
#[component]
pub fn ModelSection(#[props(into)] title: String, children: Element) -> Element {
    let heading_id = use_dom_id("models-section");
    rsx! {
        section { aria_labelledby: "{heading_id}",
            div { class: "fam-h",
                h2 { id: "{heading_id}", "{title}" }
            }
            div { class: "group surface", {children} }
        }
    }
}
