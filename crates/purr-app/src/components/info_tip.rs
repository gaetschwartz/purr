use super::icon::{Icon, IconName, IconSize};
use dioxus::prelude::*;

/// An info icon with a one-line explanation: a tooltip on hover, a line shown below on press so
/// touch screens reach it too.
#[component]
pub fn InfoTip(
    /// What the icon explains, named in its accessible label.
    #[props(into)]
    about: String,
    #[props(into)] text: String,
) -> Element {
    let mut open = use_signal(|| false);
    rsx! {
        button {
            class: if open() { "info-tip on" } else { "info-tip" },
            r#type: "button",
            title: "{text}",
            aria_label: "About {about}",
            "aria-description": "{text}",
            aria_expanded: open(),
            onclick: move |event| {
                event.stop_propagation();
                open.toggle();
            },
            Icon { name: IconName::Info, size: IconSize::Small }
        }
        if open() {
            span { class: "tip", role: "note", "{text}" }
        }
    }
}
