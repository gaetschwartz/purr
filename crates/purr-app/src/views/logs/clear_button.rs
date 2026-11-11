use super::count;
use crate::{
    components::{
        button::{Button, ButtonVariant},
        dialog::{Dialog, DialogTone},
        icon::IconName,
        toast::use_toasts,
    },
    platform::logging::get_logs_storage,
};
use dioxus::prelude::*;

/// The Clear button; it asks first, empties the capture, then calls `on_cleared`.
#[component]
pub fn ClearButton(total: usize, on_cleared: EventHandler<()>) -> Element {
    let mut asking = use_signal(|| false);
    let toasts = use_toasts();
    let message = format!(
        "Removes {} from this view. This can't be undone.",
        count::entries(total)
    );

    rsx! {
        Button {
            id: "log-clear",
            variant: ButtonVariant::Danger,
            icon: IconName::Trash,
            disabled: total == 0,
            aria_haspopup: "dialog",
            onclick: move |_| asking.set(true),
            "Clear"
        }
        if asking() {
            Dialog {
                title: "Clear the log?",
                confirm_label: "Clear log",
                tone: DialogTone::Destructive,
                on_cancel: move |()| asking.set(false),
                on_confirm: move |()| {
                    let storage = get_logs_storage();
                    let removed = storage.count();
                    storage.clear();
                    asking.set(false);
                    on_cleared.call(());
                    toasts.success("Log cleared", format!("{} removed.", count::entries(removed)));
                },
                "{message}"
            }
        }
    }
}
