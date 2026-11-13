use super::layer::{LayerFrame, LayerKind, PopoverAlign, use_layer};
use super::presence::use_closing;
use crate::utils::dom_id::use_dom_id;
use dioxus::prelude::*;

/// A glass popover anchored under the element whose id is `anchor`; a bottom sheet on phones.
///
/// Render it inside a [`Presence`](super::presence::Presence) that follows whether it is open, so it can fade out. It traps focus, closes on Escape
/// and outside clicks through `on_close`, and gives focus back to the opener when it unmounts.
/// `menu` narrows it for action menus; fill it with [`PopoverOption`] rows. `header` and `footer`
/// stay put while the children scroll between them. Handlers inside run in the popover's scope, so
/// work that outlives the popover must use `spawn_forever`.
#[component]
pub fn Popover(
    #[props(into)] anchor: String,
    #[props(into)] label: String,
    on_close: EventHandler<()>,
    #[props(default)] align: PopoverAlign,
    #[props(default)] menu: bool,
    #[props(default)] wide: bool,
    #[props(into, default)] class: String,
    header: Option<Element>,
    footer: Option<Element>,
    children: Element,
) -> Element {
    let dom_id = use_dom_id("layer");
    let closing = use_closing();
    let size = match (menu, wide) {
        (true, _) => "popover glass g3 menu",
        (false, true) => "popover glass g3 wide",
        (false, false) => "popover glass g3",
    };
    let class = if class.is_empty() {
        size.to_owned()
    } else {
        format!("{size} {class}")
    };
    let view = rsx! {
        LayerFrame {
            dom_id: dom_id.clone(),
            kind: LayerKind::Popover,
            anchor: Some(anchor),
            align,
            on_close,
            closing,
            div {
                class,
                role: "dialog",
                aria_modal: "true",
                aria_label: label,
                tabindex: "-1",
                if let Some(header) = header {
                    div { class: "phead", {header} }
                }
                div { class: "pscroll", {children} }
                if let Some(footer) = footer {
                    div { class: "pfoot", {footer} }
                }
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
    /// A tooltip for the row.
    #[props(into)]
    hint: Option<String>,
    /// Runs instead of `onselect` when the row is clicked with Ctrl or Cmd held.
    on_alt: Option<EventHandler<()>>,
    /// Runs when the pointer moves over the row; scrolling under a still pointer does not count.
    on_hover: Option<EventHandler<()>>,
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
            title: hint,
            aria_selected,
            aria_disabled: disabled.then_some("true"),
            onmousemove: move |_| {
                if let Some(hover) = on_hover {
                    hover.call(());
                }
            },
            onclick: move |event: MouseEvent| {
                if disabled {
                    return;
                }
                let modifiers = event.modifiers();
                match on_alt {
                    Some(alt) if modifiers.ctrl() || modifiers.meta() => alt.call(()),
                    Some(_) | None => onselect.call(()),
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
