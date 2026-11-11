use crate::{
    components::{
        button::{Button, ButtonVariant},
        kbd::Kbd,
        sheet::Sheet,
    },
    shortcuts::use_shortcut_listing,
    utils::primary_key::use_primary_key_label,
};
use dioxus::prelude::*;

/// The keyboard shortcuts in force on the current screen, read from the shortcut registry.
#[component]
pub fn ShortcutsSheet(on_close: EventHandler<()>) -> Element {
    let primary = use_primary_key_label();
    let listing = use_shortcut_listing();
    rsx! {
        Sheet {
            title: "Keyboard shortcuts",
            on_close,
            actions: rsx! {
                Button {
                    variant: ButtonVariant::Primary,
                    autofocus: true,
                    onclick: move |_| on_close.call(()),
                    "Done"
                }
            },
            div { class: "kbd-list",
                for info in listing {
                    Fragment { key: "{info.label}",
                        span { "{info.label}" }
                        span { class: "row", style: "gap:4px",
                            for cap in info.chord.caps(&primary.read()) {
                                Kbd { "{cap}" }
                            }
                        }
                    }
                }
                span { "Close a sheet or menu" }
                span { Kbd { "Esc" } }
            }
        }
    }
}
