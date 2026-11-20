use super::{changes::report, developer_family::FamilySelect};
use crate::{
    components::{
        button::{Button, ButtonVariant},
        group::{Group, Item},
        select::{Select, SelectOption},
        switch::Switch,
        toast::use_toasts,
    },
    settings_store::use_settings,
    utils::dom_id::use_dom_id,
};
use dioxus::prelude::*;
use purr_common::{
    developer::model::{
        BrowserModel, BrowserOrigin, Conversion, DeveloperModel, DeveloperModelId, DeveloperSource,
    },
    model::family::{ModelFamily, WhisperSize},
    web_model::{attributes::OnnxDtype, variant::TableStorage},
};
use uuid::Uuid;

const DTYPES: [OnnxDtype; 8] = [
    OnnxDtype::Fp32,
    OnnxDtype::Fp16,
    OnnxDtype::Q8,
    OnnxDtype::Int8,
    OnnxDtype::Uint8,
    OnnxDtype::Q4,
    OnnxDtype::Bnb4,
    OnnxDtype::Q4F16,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceKind {
    HuggingFace,
    LocalFiles,
}

fn dtype_options() -> Vec<SelectOption<OnnxDtype>> {
    DTYPES
        .into_iter()
        .map(|dtype| SelectOption::new(dtype, dtype.to_string()))
        .collect()
}

#[cfg(purr_web)]
async fn install_local(
    model: &DeveloperModel,
    browser: &BrowserModel,
    input_id: &str,
    mut status: Signal<Option<String>>,
) -> Result<(), String> {
    use purr_wasm::model_install::InstallProgress;
    let files = purr_wasm::developer_files::picked(input_id).map_err(|error| error.to_string())?;
    if files.is_empty() {
        return Err("Pick the model files first.".to_owned());
    }
    let mut report = |step: InstallProgress<'_>| match step {
        InstallProgress::Receiving { path, loaded_bytes } => status.set(Some(format!(
            "{path}: {}",
            crate::utils::byte_size::ByteSize(loaded_bytes)
        ))),
        InstallProgress::Installed { path } => status.set(Some(format!("{path} stored"))),
    };
    purr_wasm::developer_install::install_local(model, browser, files, &mut report)
        .await
        .map_err(|error| error.to_string())
}

#[cfg(not(purr_web))]
async fn install_local(
    _model: &DeveloperModel,
    _browser: &BrowserModel,
    _input_id: &str,
    _status: Signal<Option<String>>,
) -> Result<(), String> {
    Err("Local files can only be loaded in the browser.".to_owned())
}

/// The form that adds a browser model from a Hugging Face repository or from files on disk.
#[component]
pub fn AddBrowserModel() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let files_id = use_dom_id("developer-files");
    let mut kind = use_signal(|| SourceKind::HuggingFace);
    let mut repo = use_signal(String::new);
    let mut revision = use_signal(|| "main".to_owned());
    let mut family = use_signal(|| ModelFamily::Whisper(WhisperSize::Tiny));
    let mut encoder = use_signal(|| OnnxDtype::Fp32);
    let mut decoder = use_signal(|| OnnxDtype::Q4);
    let mut conversion = use_signal(|| Conversion::Streaming {
        table: TableStorage::Fp32,
    });
    let mut english_only = use_signal(|| false);
    let mut status = use_signal(|| None::<String>);
    let mut busy = use_signal(|| false);

    let add = {
        let files_id = files_id.clone();
        move |_| {
            let browser_kind = kind();
            let origin = match browser_kind {
                SourceKind::HuggingFace => {
                    let (repo, revision) = (repo().trim().to_owned(), revision().trim().to_owned());
                    if repo.is_empty() || revision.is_empty() {
                        toasts.error(
                            "Repository and revision needed",
                            "Name both to add a model.",
                        );
                        return;
                    }
                    BrowserOrigin::HuggingFace { repo, revision }
                }
                SourceKind::LocalFiles => BrowserOrigin::LocalFiles,
            };
            let name = match &origin {
                BrowserOrigin::HuggingFace { repo, revision } => format!("{repo} @ {revision}"),
                BrowserOrigin::LocalFiles => "Local files".to_owned(),
            };
            let browser = BrowserModel {
                origin,
                encoder_dtype: encoder(),
                decoder_dtype: decoder(),
                conversion: conversion(),
                english_only: english_only(),
            };
            let model = DeveloperModel {
                id: DeveloperModelId::from_unique(&Uuid::new_v4().simple().to_string()),
                name,
                family: family(),
                source: DeveloperSource::Browser(browser.clone()),
            };
            let files_id = files_id.clone();
            spawn(async move {
                if browser_kind == SourceKind::LocalFiles {
                    busy.set(true);
                    let stored = install_local(&model, &browser, &files_id, status).await;
                    busy.set(false);
                    status.set(None);
                    if let Err(error) = stored {
                        toasts.error("Couldn't store the model files", error);
                        return;
                    }
                }
                report(
                    toasts,
                    settings.update(|settings| settings.developer.models.push(model)),
                );
            });
        }
    };

    rsx! {
        Group {
            header: "Add a developer model",
            footer: "Dtypes name the files transformers.js asks for, such as onnx/decoder_model_merged_q4.onnx. A repository is downloaded when the model is first used.",
            div { class: "dev-form",
                div { class: "row",
                    Select {
                        label: "Source",
                        value: kind(),
                        options: vec![
                            SelectOption::new(SourceKind::HuggingFace, "Hugging Face repository"),
                            SelectOption::new(SourceKind::LocalFiles, "Files from this computer"),
                        ],
                        onchange: move |next| kind.set(next),
                    }
                }
                match kind() {
                    SourceKind::HuggingFace => rsx! {
                        label {
                            "Repository"
                            input {
                                r#type: "text",
                                value: "{repo}",
                                placeholder: "onnx-community/whisper-tiny",
                                spellcheck: "false",
                                autocomplete: "off",
                                oninput: move |event| repo.set(event.value()),
                            }
                        }
                        label {
                            "Revision (branch, tag or commit)"
                            input {
                                r#type: "text",
                                value: "{revision}",
                                spellcheck: "false",
                                autocomplete: "off",
                                oninput: move |event| revision.set(event.value()),
                            }
                        }
                    },
                    SourceKind::LocalFiles => rsx! {
                        label { r#for: files_id.clone(),
                            "Config, tokenizer and ONNX files"
                            input { id: files_id.clone(), r#type: "file", multiple: true }
                        }
                    },
                }
            }
            Item {
                title: "Family",
                control: rsx! {
                    FamilySelect { browser: true, value: family(), onchange: move |next| family.set(next) }
                },
            }
            Item {
                title: "Encoder dtype",
                control: rsx! {
                    Select {
                        label: "Encoder dtype",
                        value: encoder(),
                        options: dtype_options(),
                        onchange: move |next| encoder.set(next),
                    }
                },
            }
            Item {
                title: "Decoder dtype",
                control: rsx! {
                    Select {
                        label: "Decoder dtype",
                        value: decoder(),
                        options: dtype_options(),
                        onchange: move |next| decoder.set(next),
                    }
                },
            }
            Item {
                title: "ONNX files",
                control: rsx! {
                    Select {
                        label: "ONNX files",
                        value: conversion(),
                        options: vec![
                            SelectOption::new(
                                Conversion::Streaming { table: TableStorage::Fp32 },
                                "Convert with Purr's converter, table in fp32",
                            ),
                            SelectOption::new(
                                Conversion::Streaming { table: TableStorage::Fp16 },
                                "Convert with Purr's converter, table in fp16",
                            ),
                            SelectOption::new(Conversion::AsIs, "Use as they are"),
                        ],
                        onchange: move |next| conversion.set(next),
                    }
                },
            }
            Item {
                title: "English only",
                control: rsx! {
                    Switch {
                        label: "English only",
                        checked: english_only(),
                        onchange: move |on| english_only.set(on),
                    }
                },
            }
            Item {
                title: "Add the model",
                subtitle: status(),
                control: rsx! {
                    Button {
                        variant: ButtonVariant::Primary,
                        disabled: busy(),
                        onclick: add,
                        "Add"
                    }
                },
            }
        }
    }
}
