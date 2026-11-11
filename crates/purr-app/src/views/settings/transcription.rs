use super::{browser_panel::BrowserPanel, changes::report, sections::SettingsSection};
use crate::{
    components::{
        group::{Group, Item, TileColor},
        icon::IconName,
        language_picker::LanguagePill,
        model_picker::ModelPill,
        segmented::{Segment, Segmented},
        slider::Slider,
        switch::Switch,
        toast::use_toasts,
    },
    platform::IS_WEB,
    settings_store::use_settings,
};
use dioxus::prelude::*;
use purr_common::export_format::ExportFormat;

/// Spoken language, model, translation, decoding and export options.
#[component]
pub fn TranscriptionSection() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let (translate, temperature, export_format) = settings.with(|settings| {
        (
            settings.model.translate,
            settings.model.temperature,
            settings.ui.export_format,
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
                Item {
                    icon: IconName::Share,
                    tile: TileColor::Teal,
                    title: "Export format",
                    subtitle: "The format Export offers first.",
                    stackable: true,
                    control: rsx! {
                        Segmented {
                            label: "Export format",
                            value: export_format,
                            segments: ExportFormat::ALL
                                .into_iter()
                                .map(|format| Segment::new(format, format.short_name()))
                                .collect::<Vec<_>>(),
                            onchange: move |format| report(toasts, settings.update(|s| s.ui.export_format = format)),
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
                }
            }
            if IS_WEB {
                BrowserPanel {}
            }
        }
    }
}
