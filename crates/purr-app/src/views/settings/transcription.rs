use super::{browser_panel::BrowserPanel, changes::report, sections::SettingsSection};
use crate::{
    components::{
        group::{Group, Item, TileColor},
        icon::IconName,
        language_picker::LanguagePill,
        model_picker::ModelPill,
        slider::Slider,
        stepper::Stepper,
        switch::Switch,
        toast::use_toasts,
    },
    platform::IS_WEB,
    settings_store::use_settings,
};
use dioxus::prelude::*;

/// Spoken language, model, translation and decoding options.
#[component]
pub fn TranscriptionSection() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let (translate, temperature, max_tokens) = settings.with(|settings| {
        (
            settings.model.translate,
            settings.model.temperature,
            settings.model.max_tokens,
        )
    });
    let temperature_text = format!("{temperature:.1}");

    rsx! {
        div { class: "sect", id: SettingsSection::Transcription.anchor(),
            Group { header: SettingsSection::Transcription.label(),
                Item {
                    icon: IconName::Globe,
                    tile: TileColor::Indigo,
                    title: "Spoken language",
                    subtitle: "Automatic works for most recordings.",
                    stackable: true,
                    control: rsx! {
                        LanguagePill { id: "settings-language" }
                    },
                }
                Item {
                    icon: IconName::Cube,
                    tile: TileColor::Purple,
                    title: "Model",
                    subtitle: "Used for new transcriptions. Download and remove models in Models.",
                    stackable: true,
                    control: rsx! {
                        ModelPill { id: "settings-model" }
                    },
                }
                Item {
                    icon: IconName::Translate,
                    tile: TileColor::Orange,
                    title: "Translate to English",
                    subtitle: "Writes English text whatever language is spoken.",
                    control: rsx! {
                        Switch {
                            label: "Translate to English",
                            checked: translate,
                            onchange: move |value| report(toasts, settings.update(|s| s.model.translate = value)),
                        }
                    },
                }
                if !IS_WEB {
                    Item {
                        icon: IconName::Thermometer,
                        tile: TileColor::Pink,
                        title: "Temperature",
                        subtitle: "Lower is more predictable. Raise it if the text repeats itself.",
                        value: temperature_text.clone(),
                        Slider {
                            label: "Temperature",
                            value: temperature,
                            min: 0.0,
                            max: 1.0,
                            step: 0.1,
                            value_text: temperature_text,
                            ends: ("0".to_owned(), "1".to_owned()),
                            onchange: move |value| report(toasts, settings.update(|s| s.model.temperature = value)),
                        }
                    }
                    Item {
                        icon: IconName::Hash,
                        tile: TileColor::Gray,
                        title: "Maximum tokens",
                        subtitle: "Maximum number of tokens to generate.",
                        control: rsx! {
                            Stepper {
                                label: "Maximum tokens",
                                value: max_tokens,
                                min: 256,
                                max: 8192,
                                step: 256,
                                onchange: move |value| report(toasts, settings.update(|s| s.model.max_tokens = value)),
                            }
                        },
                    }
                }
            }
            if IS_WEB {
                BrowserPanel {}
            }
        }
    }
}
