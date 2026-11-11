use dioxus::prelude::*;

/// A small circular activity indicator; give it a `label` when nothing else says what is loading.
#[component]
pub fn Spinner(#[props(default)] large: bool, #[props(into)] label: Option<String>) -> Element {
    let class = if large { "spinner lg" } else { "spinner" };
    let hidden = label.is_none().then_some("true");
    let role = label.is_some().then_some("status");
    rsx! {
        span { class, role, aria_label: label, aria_hidden: hidden }
    }
}
