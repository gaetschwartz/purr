use super::icon::{Icon, IconName, IconSize};
use crate::transcribe::engine_label::{BackendLabel, EngineLabel};
use dioxus::prelude::*;
use purr_common::accel::AccelBackend;

const fn icon(backend: BackendLabel) -> IconName {
    match backend {
        BackendLabel::Accel(AccelBackend::Cpu) | BackendLabel::Checking => IconName::Cpu,
        BackendLabel::Accel(_) | BackendLabel::WebGpu | BackendLabel::Automatic => IconName::Bolt,
    }
}

/// The backend and hardware a transcription will run on; activating it opens the settings
/// that change them.
#[component]
pub fn EngineChip(
    #[props(into)] id: String,
    label: EngineLabel,
    onclick: EventHandler<MouseEvent>,
) -> Element {
    let checking = label.backend == BackendLabel::Checking;
    rsx! {
        button {
            id,
            class: if checking { "engine-chip checking" } else { "engine-chip" },
            title: "{label}. Change it in Settings.",
            r#type: "button",
            aria_busy: checking,
            onclick: move |event| onclick.call(event),
            Icon { name: icon(label.backend), size: IconSize::Small }
            span { class: "et",
                span { class: "eb", "{label.backend}" }
                if let Some(hardware) = &label.hardware {
                    span { class: "eh", " · {hardware}" }
                }
            }
            span { class: "sr", ", change in Settings" }
        }
    }
}
