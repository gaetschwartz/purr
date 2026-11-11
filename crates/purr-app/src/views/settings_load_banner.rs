use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonVariant},
        icon::IconName,
        toast::use_toasts,
    },
    settings_store::{status::LoadState, use_settings},
};
use dioxus::prelude::*;

/// The one notice that the stored settings could not be read, with the way out.
#[component]
pub fn SettingsLoadBanner() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let failure = match settings.load_state() {
        LoadState::Failed(failure) => failure,
        LoadState::Loading | LoadState::Loaded => return rsx! {},
    };

    rsx! {
        div { class: "content settings-load-banner",
            Banner {
                tone: BannerTone::Bad,
                title: "Your settings could not be read",
                role: "alert",
                actions: rsx! {
                    Button {
                        variant: ButtonVariant::Primary,
                        icon: IconName::Retry,
                        onclick: move |_| {
                            settings.reset_to_defaults();
                            toasts.success("Settings reset", "The defaults are saved.");
                        },
                        "Reset settings"
                    }
                },
                "Purr is running on the defaults and left the stored settings untouched. Changes are not saved until you reset them. Details: {failure}"
            }
        }
    }
}
