use super::{
    button::{Button, ButtonVariant},
    sheet::Sheet,
};
use dioxus::prelude::*;

/// Whether confirming a [`Dialog`] destroys something.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DialogTone {
    #[default]
    Default,
    Destructive,
}

/// A confirmation sheet with Cancel and a confirm button; Cancel takes focus first.
#[component]
pub fn Dialog(
    #[props(into)] title: String,
    #[props(into)] confirm_label: String,
    #[props(into, default = "Cancel".to_owned())] cancel_label: String,
    #[props(default)] tone: DialogTone,
    on_confirm: EventHandler<()>,
    on_cancel: EventHandler<()>,
    children: Element,
) -> Element {
    let confirm_variant = match tone {
        DialogTone::Default => ButtonVariant::Primary,
        DialogTone::Destructive => ButtonVariant::DangerSolid,
    };
    rsx! {
        Sheet {
            title,
            on_close: on_cancel,
            actions: rsx! {
                Button { autofocus: true, onclick: move |_| on_cancel.call(()), "{cancel_label}" }
                Button {
                    variant: confirm_variant,
                    onclick: move |_| on_confirm.call(()),
                    "{confirm_label}"
                }
            },
            p { {children} }
        }
    }
}
