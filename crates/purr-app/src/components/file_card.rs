use super::{
    button::{Button, ButtonSize},
    icon::{Icon, IconName, IconSize},
    icon_button::IconButton,
};
use dioxus::prelude::*;

/// The chosen file: icon, name, details, Replace and Remove; `problem` turns it into an error card.
#[component]
pub fn FileCard(
    #[props(into)] name: String,
    /// Duration, size and type, joined by the caller.
    #[props(into)]
    details: String,
    /// Why the file cannot be transcribed, in a sentence.
    #[props(into)]
    problem: Option<String>,
    on_replace: EventHandler<()>,
    on_remove: EventHandler<()>,
) -> Element {
    let (icon_class, icon) = if problem.is_some() {
        ("fileicon bad", IconName::Warning)
    } else {
        ("fileicon", IconName::Music)
    };
    rsx! {
        section { class: "filecard surface", aria_label: "Selected file",
            span { class: icon_class,
                Icon { name: icon, size: IconSize::Large }
            }
            div { class: "grow", style: "min-width:0",
                div { class: "name", dir: "auto", title: "{name}", "{name}" }
                div { class: "meta tnum", "{details}" }
                if let Some(problem) = &problem {
                    p { class: "file-problem", role: "alert", "{problem}" }
                }
            }
            Button {
                size: ButtonSize::Small,
                id: "replace-file",
                onclick: move |_| on_replace.call(()),
                "Replace"
            }
            IconButton {
                icon: IconName::Close,
                label: "Remove file",
                onclick: move |_| on_remove.call(()),
            }
        }
    }
}
