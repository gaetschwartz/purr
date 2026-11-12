use super::changes::report;
use crate::{
    components::{group::Item, toast::use_toasts},
    settings_store::use_settings,
    utils::dom_id::use_dom_id,
};
use dioxus::prelude::*;
use purr_common::hugging_face_token::HuggingFaceToken;

/// The token that authenticates Hugging Face downloads.
#[component]
pub fn HuggingFaceTokenItem() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let id = use_dom_id("hugging-face-token");
    let token = settings.with(|settings| {
        settings
            .model
            .hugging_face_token
            .as_ref()
            .map(|token| token.expose().to_owned())
    });

    rsx! {
        Item {
            title: "Hugging Face token",
            subtitle: "Authenticates Hugging Face downloads.",
            stackable: true,
            control: rsx! {
                label { class: "field", r#for: id.clone(),
                    input {
                        id,
                        r#type: "password",
                        value: token.unwrap_or_default(),
                        aria_label: "Hugging Face token",
                        autocomplete: "off",
                        spellcheck: "false",
                        oninput: move |event| {
                            let token = HuggingFaceToken::new(&event.value());
                            report(toasts, settings.update(|s| s.model.hugging_face_token = token));
                        },
                    }
                }
            },
        }
    }
}
