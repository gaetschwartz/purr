use super::icon::{Icon, IconName};
use dioxus::prelude::*;

/// Meaning and colour of a [`Banner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BannerTone {
    #[default]
    Info,
    Ok,
    Warn,
    Bad,
}

impl BannerTone {
    const fn class(self) -> &'static str {
        match self {
            Self::Info => "banner info",
            Self::Ok => "banner ok",
            Self::Warn => "banner warn",
            Self::Bad => "banner bad",
        }
    }

    /// The glyph shown when the banner does not choose one.
    #[must_use]
    pub const fn default_icon(self) -> IconName {
        match self {
            Self::Info => IconName::Info,
            Self::Ok => IconName::CheckCircle,
            Self::Warn => IconName::Warning,
            Self::Bad => IconName::XCircle,
        }
    }
}

/// An inline message with an icon, a bold title, a body and optional action buttons.
///
/// Pass `role: "alert"` for errors that appear after an action, `role: "status"` for progress news.
#[component]
pub fn Banner(
    #[props(default)] tone: BannerTone,
    icon: Option<IconName>,
    #[props(into)] title: String,
    actions: Option<Element>,
    #[props(extends = GlobalAttributes)] attributes: Vec<Attribute>,
    children: Element,
) -> Element {
    let has_body = children != VNode::empty();
    rsx! {
        div { class: tone.class(), ..attributes,
            Icon { name: icon.unwrap_or_else(|| tone.default_icon()) }
            div { class: "grow",
                b { "{title}" }
                if has_body {
                    p { {children} }
                }
                if let Some(actions) = actions {
                    div { class: "acts", {actions} }
                }
            }
        }
    }
}
