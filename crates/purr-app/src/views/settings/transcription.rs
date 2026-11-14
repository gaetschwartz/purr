use super::{browser_panel::BrowserPanel, hugging_face_token::HuggingFaceTokenItem};
use crate::{
    components::{
        group::{Group, Item},
        language_picker::LanguagePill,
        model_picker::ModelPill,
        pick_target::PickTarget,
    },
    platform::IS_WEB,
};
use dioxus::prelude::*;

/// Spoken language, model and download options.
#[component]
pub fn TranscriptionPage() -> Element {
    rsx! {
        Group {
            Item {
                title: "Default language",
                stackable: true,
                control: rsx! {
                    LanguagePill { id: "settings-language", target: PickTarget::Default }
                },
            }
            Item {
                title: "Default model",
                stackable: true,
                control: rsx! {
                    ModelPill { id: "settings-model", target: PickTarget::Default }
                },
            }
            HuggingFaceTokenItem {}
        }
        if IS_WEB {
            BrowserPanel {}
        }
    }
}
