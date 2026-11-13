use super::{
    browser_panel::BrowserPanel, changes::report, hugging_face_token::HuggingFaceTokenItem,
    sections::SettingsSection,
};
use crate::{
    components::{
        group::{Group, Item},
        language_picker::LanguagePill,
        model_picker::ModelPill,
        pick_target::PickTarget,
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
                Item {
                    title: "Translate to English",
                    control: rsx! {
                        Switch {
                            label: "Translate to English",
                            checked: translate,
                            onchange: move |value| report(toasts, settings.update(|s| s.model.translate = value)),
                        }
                    },
                }
                Item {
                    title: "Default export format",
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
                        title: "Temperature",
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
