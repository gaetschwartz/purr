use super::icon::{Icon, IconName, IconSize};
use crate::{main_app::Route, utils::dom_id::use_dom_id};
use dioxus::prelude::*;

/// Background of the small icon tile at the start of an [`Item`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TileColor {
    #[default]
    Indigo,
    Blue,
    Teal,
    Orange,
    Green,
    Gray,
    Pink,
    Purple,
    Red,
}

impl TileColor {
    const fn class(self) -> &'static str {
        match self {
            Self::Indigo => "ic indigo",
            Self::Blue => "ic blue",
            Self::Teal => "ic teal",
            Self::Orange => "ic orange",
            Self::Green => "ic green",
            Self::Gray => "ic gray",
            Self::Pink => "ic pink",
            Self::Purple => "ic purple",
            Self::Red => "ic red",
        }
    }
}

/// An inset grouped list on glass, with an optional caps header and a footnote.
#[component]
pub fn Group(
    #[props(into)] header: Option<String>,
    #[props(into)] footer: Option<String>,
    #[props(into)] label: Option<String>,
    #[props(extends = GlobalAttributes)] attributes: Vec<Attribute>,
    children: Element,
) -> Element {
    let header_id = use_dom_id("group");
    let labelled_by = header.as_ref().map(|_| header_id.clone());
    rsx! {
        section { aria_labelledby: labelled_by, aria_label: label, ..attributes,
            if let Some(header) = header {
                h2 { class: "group-h", id: header_id, "{header}" }
            }
            div { class: "group glass", {children} }
            if let Some(footer) = footer {
                p { class: "group-f", "{footer}" }
            }
        }
    }
}

/// What activating an [`Item`] does.
#[derive(Clone, PartialEq)]
pub enum ItemAction {
    Navigate(Route),
    Press(EventHandler<MouseEvent>),
}

/// A row of a [`Group`]: icon tile, title, subtitle and a trailing value or control.
///
/// With an `action` the whole row is a link or a button and shows a chevron; `control` holds a
/// switch, select or stepper (name it with the same words as the title); `children` render under
/// the row, indented past the tile, for wide controls such as a slider.
#[component]
pub fn Item(
    #[props(into)] title: String,
    #[props(into)] subtitle: Option<String>,
    icon: Option<IconName>,
    #[props(default)] tile: TileColor,
    #[props(into)] value: Option<String>,
    control: Option<Element>,
    action: Option<ItemAction>,
    #[props(default)] stackable: bool,
    #[props(default)] danger: bool,
    #[props(extends = GlobalAttributes)] attributes: Vec<Attribute>,
    children: Element,
) -> Element {
    let has_body = children != VNode::empty();
    let mut class = String::from("item");
    if stackable {
        class.push_str(" stackable");
    }
    if danger {
        class.push_str(" danger");
    }
    if has_body {
        class.push_str(" col");
    }
    let chevron = action.is_some();
    let head = rsx! {
        if let Some(name) = icon {
            span { class: tile.class(),
                Icon { name }
            }
        }
        span { class: "txt",
            b { "{title}" }
            if let Some(subtitle) = &subtitle {
                span { "{subtitle}" }
            }
        }
        if let Some(value) = &value {
            span { class: "val", "{value}" }
        }
        if let Some(control) = control {
            span { class: "ctl", {control} }
        }
        if chevron {
            Icon {
                name: IconName::ChevronRight,
                size: IconSize::Small,
                class: "faint",
            }
        }
    };
    let inner = if has_body {
        rsx! {
            div { class: "head", {head} }
            div { class: "body", {children} }
        }
    } else {
        head
    };
    match action {
        Some(ItemAction::Navigate(to)) => rsx! {
            Link { class, to, attributes, {inner} }
        },
        Some(ItemAction::Press(handler)) => rsx! {
            button {
                class,
                r#type: "button",
                onclick: move |event| handler.call(event),
                ..attributes,
                {inner}
            }
        },
        None => rsx! {
            div { class, ..attributes, {inner} }
        },
    }
}
