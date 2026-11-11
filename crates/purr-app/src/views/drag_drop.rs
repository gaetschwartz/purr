use crate::{
    components::upload_zone::{UploadState, UploadZone},
    main_app::{Route, UploadNames},
    platform,
    utils::file_kind::may_contain_audio,
};
use bytes::Bytes;
use dioxus::{html::FileData, html::HasFileData, prelude::*};
use purr_common::platform::{FileId, Platform as _};
use tracing::{error, info};

/// Why a chosen file was turned away.
#[derive(Debug, thiserror::Error)]
enum UploadError {
    #[error("It is a {content_type} file, not audio.")]
    NotAudio { content_type: String },
    #[error("The browser could not read the file: {0}")]
    Unreadable(String),
    #[error("{0}")]
    Rejected(String),
}

async fn store_file(file: &FileData) -> Result<FileId, UploadError> {
    let content_type = file.content_type().unwrap_or_default();
    if !may_contain_audio(&content_type) {
        return Err(UploadError::NotAudio { content_type });
    }
    let bytes = file
        .read_bytes()
        .await
        .map_err(|e| UploadError::Unreadable(e.to_string()))?;
    info!("Read {} ({} bytes)", file.name(), bytes.len());
    platform::get_platform()
        .process_file(Bytes::from(bytes.to_vec()), file.name().as_ref())
        .await
        .map_err(|e| UploadError::Rejected(e.to_string()))
}

#[component]
pub fn DragDropZone() -> Element {
    let mut is_dragging = use_signal(|| false);
    let mut reading = use_signal(|| None::<String>);
    let mut failure = use_signal(|| None::<(String, String)>);
    let navigator = use_navigator();
    let names = use_context::<UploadNames>();

    let upload_state = match (reading(), failure(), is_dragging()) {
        (Some(file_name), _, _) => UploadState::Reading { file_name },
        (None, Some((file_name, message)), _) => UploadState::Failed { file_name, message },
        (None, None, true) => UploadState::Dragging,
        (None, None, false) => UploadState::Idle,
    };

    let handle_file = move |file: FileData| async move {
        let file_name = file.name();
        failure.set(None);
        reading.set(Some(file_name.clone()));
        let outcome = store_file(&file).await;
        reading.set(None);
        match outcome {
            Ok(file_id) => {
                let mut known = names.0;
                known.write().insert(file_id.clone(), file_name);
                navigator.push(Route::Transcription { file: file_id });
            }
            Err(e) => {
                error!("Could not use {file_name}: {e}");
                failure.set(Some((file_name, e.to_string())));
            }
        }
    };

    rsx! {
        UploadZone {
            state: upload_state,
            ondragover: move |evt: DragEvent| {
                evt.prevent_default();
                is_dragging.set(true);
            },
            ondragleave: move |_evt: DragEvent| is_dragging.set(false),
            ondrop: move |evt: DragEvent| async move {
                evt.prevent_default();
                is_dragging.set(false);
                if let Some(file) = evt.data().files().into_iter().next() {
                    handle_file(file).await;
                }
            },
            onchange: move |evt: FormEvent| async move {
                if let Some(file) = evt.files().into_iter().next() {
                    handle_file(file).await;
                }
            },
        }
    }
}
