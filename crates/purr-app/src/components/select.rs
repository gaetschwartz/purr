use super::icon::{Icon, IconName, IconSize};
use dioxus::prelude::*;

/// One entry of a [`Select`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectOption<T> {
    pub value: T,
    pub label: String,
    pub disabled: bool,
}

impl<T> SelectOption<T> {
    /// An enabled option.
    pub fn new(value: T, label: impl Into<String>) -> Self {
        Self {
            value,
            label: label.into(),
            disabled: false,
        }
    }

    /// Greys the option out.
    #[must_use]
    pub const fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
}

/// A native `<select>` drawn as the Apple pop-up button; values stay typed.
#[component]
pub fn Select<T: Clone + PartialEq + 'static>(
    options: Vec<SelectOption<T>>,
    value: T,
    onchange: EventHandler<T>,
    #[props(into)] label: Option<String>,
    #[props(into)] labelled_by: Option<String>,
    #[props(default)] disabled: bool,
    #[props(extends = GlobalAttributes)] attributes: Vec<Attribute>,
) -> Element {
    let selected = options.iter().position(|option| option.value == value);
    let choices = options.clone();
    rsx! {
        span { class: "select",
            select {
                aria_label: label,
                aria_labelledby: labelled_by,
                disabled,
                onchange: move |event| {
                    let chosen = event
                        .value()
                        .parse::<usize>()
                        .ok()
                        .and_then(|index| choices.get(index));
                    if let Some(option) = chosen {
                        onchange.call(option.value.clone());
                    } else {
                        tracing::warn!("The select reported an unknown option: {}", event.value());
                    }
                },
                ..attributes,
                for (index , entry) in options.iter().enumerate() {
                    option {
                        key: "{index}",
                        value: "{index}",
                        selected: selected == Some(index),
                        disabled: entry.disabled,
                        "{entry.label}"
                    }
                }
            }
            Icon { name: IconName::UpDown, size: IconSize::Small }
        }
    }
}
