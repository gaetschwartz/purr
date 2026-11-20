use super::{changes::report, developer_family::FamilySelect};
use crate::{
    components::{
        button::{Button, ButtonVariant},
        group::{Group, Item},
        toast::use_toasts,
    },
    settings_store::use_settings,
};
use dioxus::prelude::*;
use purr_common::{
    developer::model::{DeveloperModel, DeveloperModelId, DeveloperSource, NativeModel},
    model::{
        family::{ModelFamily, WhisperSize},
        format::ModelFormat,
    },
};
use std::path::PathBuf;
use uuid::Uuid;

#[cfg(feature = "desktop")]
async fn pick_file() -> Option<PathBuf> {
    rfd::AsyncFileDialog::new()
        .set_title("Choose a GGML or GGUF model file")
        .pick_file()
        .await
        .map(|handle| handle.path().to_path_buf())
}

#[cfg(not(feature = "desktop"))]
async fn pick_file() -> Option<PathBuf> {
    None
}

#[cfg(feature = "desktop")]
async fn read_format(path: &std::path::Path) -> Result<ModelFormat, String> {
    use tokio::io::AsyncReadExt as _;
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let mut header = [0_u8; 4];
    file.read_exact(&mut header)
        .await
        .map_err(|error| format!("{}: {error}", path.display()))?;
    ModelFormat::of_header(&header)
        .ok_or_else(|| format!("{} is neither a GGML nor a GGUF file.", path.display()))
}

#[cfg(not(feature = "desktop"))]
async fn read_format(_path: &std::path::Path) -> Result<ModelFormat, String> {
    Err("Model files can only be loaded on the desktop.".to_owned())
}

/// The form that adds a model from a GGML or GGUF file on disk; the engine follows the format.
#[component]
pub fn AddNativeModel() -> Element {
    let settings = use_settings();
    let toasts = use_toasts();
    let mut family = use_signal(|| ModelFamily::Whisper(WhisperSize::Tiny));

    let choose = move |_| {
        let chosen_family = family();
        spawn(async move {
            let Some(path) = pick_file().await else {
                return;
            };
            let format = match read_format(&path).await {
                Ok(format) => format,
                Err(error) => {
                    toasts.error("That file can't be loaded", error);
                    return;
                }
            };
            let name = path.file_name().map_or_else(
                || path.display().to_string(),
                |name| name.to_string_lossy().into_owned(),
            );
            let model = DeveloperModel {
                id: DeveloperModelId::from_unique(&Uuid::new_v4().simple().to_string()),
                name,
                family: chosen_family,
                source: DeveloperSource::Native(NativeModel { path, format }),
            };
            report(
                toasts,
                settings.update(|settings| settings.developer.models.push(model)),
            );
        });
    };

    rsx! {
        Group {
            header: "Add a developer model",
            footer: "The model is read from where it is. A GGML file runs on the whisper.cpp engine and a GGUF file on the transcribe.cpp engine.",
            Item {
                title: "Family",
                control: rsx! {
                    FamilySelect { browser: false, value: family(), onchange: move |next| family.set(next) }
                },
            }
            Item {
                title: "Model file",
                control: rsx! {
                    Button { variant: ButtonVariant::Primary, onclick: choose, "Choose file…" }
                },
            }
        }
    }
}
