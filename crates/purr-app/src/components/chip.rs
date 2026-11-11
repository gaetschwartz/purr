use super::icon::{Icon, IconName};
use dioxus::prelude::*;

/// Colour of a [`Chip`]; semantic tones always come with words, never colour alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChipTone {
    #[default]
    Neutral,
    Accent,
    Ok,
    Warn,
    Bad,
}

impl ChipTone {
    const fn class(self) -> &'static str {
        match self {
            Self::Neutral => "chip",
            Self::Accent => "chip accent",
            Self::Ok => "chip ok",
            Self::Warn => "chip warn",
            Self::Bad => "chip bad",
        }
    }
}

/// A small rounded status label with an optional icon.
#[component]
pub fn Chip(
    #[props(default)] tone: ChipTone,
    icon: Option<IconName>,
    #[props(extends = GlobalAttributes)] attributes: Vec<Attribute>,
    children: Element,
) -> Element {
    rsx! {
        span { class: tone.class(), ..attributes,
            if let Some(name) = icon {
                Icon { name }
            }
            {children}
        }
    }
}
