use super::layer::{LayerFrame, LayerKind, PopoverAlign, use_layer};
use crate::utils::dom_id::use_dom_id;
use dioxus::prelude::*;

/// A glass popover anchored under the element whose id is `anchor`; a bottom sheet on phones.
///
/// Render it only while open (`if open() { Popover { .. } }`). It traps focus, closes on Escape
/// and outside clicks through `on_close`, and gives focus back to the opener when it unmounts.
/// `menu` narrows it for action menus; fill it with [`PopoverOption`] rows. Handlers inside run in
/// the popover's scope, so work that outlives the popover must use `spawn_forever`.
#[component]
pub fn Popover(
    #[props(into)] anchor: String,
    #[props(into)] label: String,
    on_close: EventHandler<()>,
    #[props(default)] align: PopoverAlign,
    #[props(default)] menu: bool,
    #[props(default)] wide: bool,
    children: Element,
) -> Element {
    let dom_id = use_dom_id("layer");
    let class = match (menu, wide) {
        (true, _) => "popover glass g3 menu",
        (false, true) => "popover glass g3 wide",
        (false, false) => "popover glass g3",
    };
    let view = rsx! {
        LayerFrame {
            dom_id: dom_id.clone(),
            kind: LayerKind::Popover,
            anchor: Some(anchor),
            align,
            on_close,
            div {
                class,
                role: "dialog",
                aria_modal: "true",
                aria_label: label,
                tabindex: "-1",
                div { class: "pscroll", {children} }
            }
        }
    };
    use_layer(&dom_id, view);
    rsx! {}
}

/// The ARIA role of a [`PopoverOption`]: a command in a menu or a choice in a list box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptionRole {
    #[default]
    MenuItem,
    Option,
}

/// A row of a popover: leading check or icon, title, subtitle and trailing text.
#[component]
pub fn PopoverOption(
    #[props(into)] title: String,
    #[props(into)] subtitle: Option<String>,
    #[props(into)] trailing: Option<String>,
    /// Words read before `trailing` by screen readers only.
    #[props(into)]
    trailing_prefix: Option<String>,
    leading: Option<Element>,
    /// A mark after the trailing text, such as a downloaded badge.
    status: Option<Element>,
    #[props(default)] role: OptionRole,
    #[props(default)] selected: bool,
    #[props(default)] disabled: bool,
    #[props(default)] danger: bool,
    /// Marks the row the keyboard points at while focus stays in a search field.
    #[props(default)]
    highlighted: bool,
    #[props(into)] id: Option<String>,
    onselect: EventHandler<()>,
) -> Element {
    let class = match (danger, highlighted) {
        (true, true) => "opt danger kb",
        (true, false) => "opt danger",
        (false, true) => "opt kb",
        (false, false) => "opt",
    };
    let (role, aria_selected) = match role {
        OptionRole::MenuItem => ("menuitem", None),
        OptionRole::Option => ("option", Some(selected)),
    };
    rsx! {
        button {
            id,
            class,
            r#type: "button",
            role,
            aria_selected,
            aria_disabled: disabled.then_some("true"),
            onclick: move |_| {
                if !disabled {
                    onselect.call(());
                }
            },
            span { class: "ck", {leading} }
            span { class: "t",
                b { "{title}" }
                if let Some(subtitle) = &subtitle {
                    span { "{subtitle}" }
                }
            }
            if let Some(trailing) = &trailing {
                span { class: "r",
                    if let Some(prefix) = &trailing_prefix {
                        span { class: "sr", "{prefix} " }
                    }
                    "{trailing}"
                }
            }
            if let Some(status) = status {
                span { class: "st", {status} }
            }
        }
    }
}
