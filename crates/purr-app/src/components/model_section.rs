use crate::{model_catalog::credit::Credit, utils::dom_id::use_dom_id};
use dioxus::prelude::*;

/// A titled list section of the Models screen: a small caps header, then a grouped list, then the
/// credit of the models' makers when their licence asks for one.
#[component]
pub fn ModelSection(
    #[props(into)] title: String,
    #[props(default)] credit: Option<Credit>,
    children: Element,
) -> Element {
    let heading_id = use_dom_id("models-section");
    rsx! {
        section { aria_labelledby: "{heading_id}",
            div { class: "fam-h",
                h2 { id: "{heading_id}", "{title}" }
            }
            div { class: "group surface", {children} }
            if let Some(credit) = credit {
                p { class: "fam-note",
                    "{credit.line}"
                    if let Some(changes) = credit.changes {
                        " {changes}"
                    }
                    " Source: {credit.origin_url}. Licence: {credit.license_url}."
                }
            }
        }
    }
}
