use super::icon::{Icon, IconName, IconSize};
use dioxus::prelude::*;
use std::rc::Rc;

/// One choice of a [`Segmented`] control.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment<T> {
    pub value: T,
    pub label: String,
    pub icon: Option<IconName>,
}

impl<T> Segment<T> {
    /// A text-only segment.
    pub fn new(value: T, label: impl Into<String>) -> Self {
        Self {
            value,
            label: label.into(),
            icon: None,
        }
    }

    /// Adds a leading icon.
    #[must_use]
    pub const fn with_icon(mut self, icon: IconName) -> Self {
        self.icon = Some(icon);
        self
    }
}

/// The segment the arrow keys move to, wrapping at both ends.
const fn step(current: usize, len: usize, forward: bool) -> usize {
    if len == 0 {
        0
    } else if forward {
        (current + 1) % len
    } else {
        (current + len - 1) % len
    }
}

/// An Apple segmented control: a radio group moved with the arrow keys, `full` stretches it.
#[component]
pub fn Segmented<T: Clone + PartialEq + 'static>(
    segments: Vec<Segment<T>>,
    value: T,
    onchange: EventHandler<T>,
    #[props(into)] label: String,
    #[props(default)] full: bool,
    #[props(default)] disabled: bool,
) -> Element {
    let mut mounted = use_signal(Vec::<Option<Rc<MountedData>>>::new);
    let count = segments.len();
    let selected = segments
        .iter()
        .position(|segment| segment.value == value)
        .unwrap_or_default();
    let class = if full { "seg full" } else { "seg" };
    let choices = segments.clone();

    let onkeydown = move |event: KeyboardEvent| {
        let forward = match event.key() {
            Key::Named(NamedKey::ArrowRight | NamedKey::ArrowDown) => true,
            Key::Named(NamedKey::ArrowLeft | NamedKey::ArrowUp) => false,
            Key::Named(_) | Key::Character(_) => return,
        };
        event.prevent_default();
        let next = step(selected, count, forward);
        if let Some(segment) = choices.get(next) {
            onchange.call(segment.value.clone());
        }
        if let Some(Some(element)) = mounted.read().get(next).cloned() {
            spawn(async move {
                if let Err(error) = element.set_focus(true).await {
                    tracing::debug!("Could not focus the segment: {error}");
                }
            });
        }
    };

    rsx! {
        div {
            class,
            role: "radiogroup",
            aria_label: label,
            aria_disabled: disabled.then_some("true"),
            onkeydown,
            for (index , segment) in segments.into_iter().enumerate() {
                button {
                    key: "{index}",
                    r#type: "button",
                    role: "radio",
                    aria_checked: index == selected,
                    tabindex: if index == selected { "0" } else { "-1" },
                    disabled,
                    onmounted: move |event: MountedEvent| {
                        let mut list = mounted.write();
                        if list.len() <= index {
                            list.resize(index + 1, None);
                        }
                        if let Some(slot) = list.get_mut(index) {
                            *slot = Some(event.data());
                        }
                    },
                    onclick: {
                        let value = segment.value;
                        move |_| onchange.call(value.clone())
                    },
                    if let Some(name) = segment.icon {
                        Icon { name, size: IconSize::Small }
                    }
                    "{segment.label}"
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::step;

    #[test]
    fn arrow_keys_wrap_around_the_segments() {
        assert_eq!(step(0, 3, true), 1);
        assert_eq!(step(2, 3, true), 0);
        assert_eq!(step(0, 3, false), 2);
        assert_eq!(step(0, 0, true), 0);
    }
}
