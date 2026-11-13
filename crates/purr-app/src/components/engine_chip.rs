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

/// The backend a transcription will run on; the hardware name is in the tooltip.
#[component]
pub fn EngineChip(#[props(into)] id: String, label: EngineLabel) -> Element {
    let checking = label.backend == BackendLabel::Checking;
    rsx! {
        span {
            id,
            class: if checking { "engine-chip checking" } else { "engine-chip" },
            title: "{label}",
            aria_busy: checking,
            Icon { name: icon(label.backend), size: IconSize::Small }
            span { "{label.backend}" }
            if let Some(hardware) = &label.hardware {
                span { class: "sr", ", {hardware}" }
            }
        }
    }
}
