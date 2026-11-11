use dioxus::prelude::*;

/// A keyboard key cap, e.g. `Kbd { "Ctrl+O" }`.
#[component]
pub fn Kbd(children: Element) -> Element {
    rsx! {
        kbd { {children} }
    }
}
