use super::icon::{Icon, IconName, IconSize};
use dioxus::prelude::*;

/// Visual weight of a [`Button`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ButtonVariant {
    Primary,
    #[default]
    Secondary,
    /// Tinted with the accent: the one most common action of a view.
    Tonal,
    Plain,
    Danger,
    DangerSolid,
}

impl ButtonVariant {
    const fn class(self) -> &'static str {
        match self {
            Self::Primary => " primary",
            Self::Secondary => "",
            Self::Tonal => " tonal",
            Self::Plain => " plain",
            Self::Danger => " danger",
            Self::DangerSolid => " danger solid",
        }
    }
}

/// Height of a [`Button`]: 30, 36 or 44 px.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ButtonSize {
    Small,
    #[default]
    Medium,
    Large,
}

impl ButtonSize {
    const fn class(self) -> &'static str {
        match self {
            Self::Small => " sm",
            Self::Medium => "",
            Self::Large => " lg",
        }
    }

    const fn icon(self) -> IconSize {
        match self {
            Self::Small | Self::Medium => IconSize::Small,
            Self::Large => IconSize::Medium,
        }
    }
}

/// Whether a [`Button`] submits its form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ButtonType {
    #[default]
    Button,
    Submit,
}

impl ButtonType {
    const fn attribute(self) -> &'static str {
        match self {
            Self::Button => "button",
            Self::Submit => "submit",
        }
    }
}

/// A pill-shaped push button with an optional leading icon.
#[component]
pub fn Button(
    #[props(default)] variant: ButtonVariant,
    #[props(default)] size: ButtonSize,
    #[props(default)] kind: ButtonType,
    icon: Option<IconName>,
    #[props(default)] disabled: bool,
    onclick: Option<EventHandler<MouseEvent>>,
    #[props(extends = GlobalAttributes)] attributes: Vec<Attribute>,
    children: Element,
) -> Element {
    rsx! {
        button {
            class: "btn{variant.class()}{size.class()}",
            r#type: kind.attribute(),
            disabled,
            onclick: move |event| {
                if let Some(handler) = onclick {
                    handler.call(event);
                }
            },
            ..attributes,
            if let Some(name) = icon {
                Icon { name, size: size.icon() }
            }
            {children}
        }
    }
}
