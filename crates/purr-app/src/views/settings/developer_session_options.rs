use super::changes::report;
use crate::{
    components::{group::Group, toast::use_toasts},
    settings_store::use_settings,
    utils::dom_id::use_dom_id,
};
use dioxus::prelude::*;
use purr_common::developer::session_options::parse;

const HINT: &str = r#"A JSON object merged into the options the browser builds, such as {"executionProviders":[{"name":"webgpu","storageBufferCacheMode":"simple"}]}. Applies from the next model load."#;

/// ONNX Runtime session options as JSON, merged into the ones the browser builds.
#[component]
pub fn SessionOptionsGroup() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let id = use_dom_id("session-options");
    let saved = settings.with(|settings| settings.developer.session_options.clone());
    let mut draft = use_signal(|| saved.clone().unwrap_or_default());
    let problem = parse(&draft.read()).err().map(|error| error.to_string());

    rsx! {
        Group {
            header: "ONNX Runtime session options",
            footer: HINT,
            div { class: "dev-form",
                label { r#for: id.clone(),
                    "Session options"
                    textarea {
                        id,
                        value: "{draft}",
                        spellcheck: "false",
                        autocomplete: "off",
                        aria_invalid: problem.is_some().then_some("true"),
                        oninput: move |event| draft.set(event.value()),
                        onchange: move |event| {
                            let text = event.value();
                            if parse(&text).is_ok() {
                                let stored = (!text.trim().is_empty()).then_some(text);
                                report(
                                    toasts,
                                    settings.update(|settings| settings.developer.session_options = stored),
                                );
                            }
                        },
                    }
                }
                if let Some(problem) = &problem {
                    p { class: "group-f", role: "alert", "{problem}" }
                }
            }
        }
    }
}
