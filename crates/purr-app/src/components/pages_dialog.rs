use super::{
    icon::IconName,
    icon_button::IconButton,
    layer::{LayerFrame, LayerKind, use_layer},
};
use crate::utils::dom_id::use_dom_id;
use dioxus::prelude::*;

/// A large modal dialog: a header, a list of pages on the left (above the content on phones) and
/// the selected page's content, which scrolls on its own.
///
/// Render it only while open. Like [`super::sheet::Sheet`] it is labelled by its `title`, traps
/// focus, closes on Escape and scrim clicks through `on_close`, and returns focus to the opener.
/// Put an `autofocus` attribute on the control in `nav` that should take focus first.
#[component]
pub fn PagesDialog(
    #[props(into)] title: String,
    on_close: EventHandler<()>,
    actions: Option<Element>,
    nav: Element,
    children: Element,
) -> Element {
    let dom_id = use_dom_id("layer");
    let title_id = format!("{dom_id}-title");
    let view = rsx! {
        LayerFrame {
            dom_id: dom_id.clone(),
            kind: LayerKind::Sheet,
            anchor: None,
            on_close,
            div { class: "pdlg-wrap",
                div {
                    class: "pdlg glass g3",
                    role: "dialog",
                    aria_modal: "true",
                    aria_labelledby: title_id.clone(),
                    tabindex: "-1",
                    div { class: "pdlg-head",
                        h2 { id: title_id, "{title}" }
                        {actions}
                        IconButton {
                            icon: IconName::Close,
                            label: "Close",
                            onclick: move |_| on_close.call(()),
                        }
                    }
                    div { class: "pdlg-main",
                        {nav}
                        div { class: "pdlg-scroll", {children} }
                    }
                }
            }
        }
    };
    use_layer(&dom_id, view);
    rsx! {}
}
