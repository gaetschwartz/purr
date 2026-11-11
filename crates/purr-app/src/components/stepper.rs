use super::icon::{Icon, IconName, IconSize};
use dioxus::prelude::*;

/// The value one press of a stepper button leads to, kept within `min..=max`.
const fn stepped(value: u32, step: u32, min: u32, max: u32, up: bool) -> u32 {
    let next = if up {
        value.saturating_add(step)
    } else {
        value.saturating_sub(step)
    };
    if next < min {
        min
    } else if next > max {
        max
    } else {
        next
    }
}

/// A minus / value / plus control for small integers; `display` replaces the number, e.g. "Auto".
#[component]
pub fn Stepper(
    value: u32,
    min: u32,
    max: u32,
    #[props(default = 1)] step: u32,
    #[props(into)] label: String,
    #[props(into)] display: Option<String>,
    #[props(default)] disabled: bool,
    onchange: EventHandler<u32>,
) -> Element {
    let shown = display.unwrap_or_else(|| value.to_string());
    rsx! {
        div { class: "stepper", role: "group", aria_label: label.clone(),
            button {
                r#type: "button",
                aria_label: "Decrease {label}",
                disabled: disabled || value <= min,
                onclick: move |_| onchange.call(stepped(value, step, min, max, false)),
                Icon { name: IconName::Minus, size: IconSize::Small }
            }
            output { class: "tnum", aria_live: "polite", "{shown}" }
            button {
                r#type: "button",
                aria_label: "Increase {label}",
                disabled: disabled || value >= max,
                onclick: move |_| onchange.call(stepped(value, step, min, max, true)),
                Icon { name: IconName::Plus, size: IconSize::Small }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::stepped;

    #[test]
    fn steps_stay_within_the_limits() {
        assert_eq!(stepped(224, 32, 32, 448, true), 256);
        assert_eq!(stepped(440, 32, 32, 448, true), 448);
        assert_eq!(stepped(40, 32, 32, 448, false), 32);
        assert_eq!(stepped(0, 1, 0, 24, false), 0);
    }
}
