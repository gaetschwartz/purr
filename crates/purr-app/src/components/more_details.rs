use dioxus::prelude::*;

/// A closed "Details" disclosure for the causes behind a short message: paths, libraries, raw errors.
#[component]
pub fn MoreDetails(children: Element) -> Element {
    rsx! {
        details { class: "more",
            summary { "Details" }
            div { class: "more-body", {children} }
        }
    }
}
