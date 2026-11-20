use crate::components::select::{Select, SelectOption};
use dioxus::prelude::*;
use purr_common::model::family::{ModelFamily, Qwen3AsrSize, WhisperSize};
use strum::VariantArray as _;

/// The families a developer model can be grouped under, in listing order.
fn families(browser: bool) -> Vec<ModelFamily> {
    if browser {
        return WhisperSize::VARIANTS
            .iter()
            .copied()
            .map(ModelFamily::Whisper)
            .collect();
    }
    ModelFamily::all()
        .chain([
            ModelFamily::Qwen3Asr(Qwen3AsrSize::B0_6),
            ModelFamily::Qwen3Asr(Qwen3AsrSize::B1_7),
        ])
        .collect()
}

/// The family that groups the model in the picker.
#[component]
pub fn FamilySelect(
    browser: bool,
    value: ModelFamily,
    onchange: EventHandler<ModelFamily>,
) -> Element {
    let options = families(browser)
        .into_iter()
        .map(|family| SelectOption::new(family, family.label()))
        .collect();
    rsx! {
        Select {
            label: "Family",
            value,
            options,
            onchange,
        }
    }
}
