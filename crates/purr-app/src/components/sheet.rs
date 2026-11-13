use super::layer::{LayerFrame, LayerKind, use_layer};
use crate::utils::dom_id::use_dom_id;
use dioxus::prelude::*;

/// A modal glass sheet centred over the app (pinned to the bottom on phones).
///
/// Render it only while open. It is labelled by its `title`, traps focus, closes on Escape and
/// scrim clicks through `on_close`, and returns focus to the opener. Put an `autofocus` attribute
/// on the control that should take focus first; otherwise the first focusable one does. Handlers
/// inside run in the sheet's scope, so work that outlives the sheet must use `spawn_forever`.
#[component]
pub fn Sheet(
    #[props(into)] title: String,
    on_close: EventHandler<()>,
    #[props(default)] wide: bool,
    actions: Option<Element>,
    children: Element,
) -> Element {
    let dom_id = use_dom_id("layer");
    let title_id = format!("{dom_id}-title");
    let class = if wide {
        "sheet glass g3 wide"
    } else {
        "sheet glass g3"
    };
    let view = rsx! {
        LayerFrame {
            dom_id: dom_id.clone(),
            kind: LayerKind::Sheet,
            anchor: None,
            on_close,
            div { class: "sheet-wrap",
                div {
                    class,
                    role: "dialog",
                    aria_modal: "true",
                    aria_labelledby: title_id.clone(),
                    tabindex: "-1",
                    div { class: "sscroll",
                        h2 { id: title_id, "{title}" }
                        {children}
                        if let Some(actions) = actions {
                            div { class: "acts", {actions} }
                        }
                    }
                }
            }
        }
    };
    use_layer(&dom_id, view);
    rsx! {}
}
