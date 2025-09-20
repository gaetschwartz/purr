use crate::{
    components::{UploadState, UploadZone},
    platform, Route,
};
use bytes::Bytes;
use dioxus::{
    html::{FileEngine, HasFileData},
    prelude::*,
};
use std::sync::Arc;
use tracing::{error, info};

#[component]
pub fn DragDropZone() -> Element {
    let mut is_dragging = use_signal(|| false);
    let mut uploaded_file = use_signal(|| None::<String>);
    let mut is_uploading = use_signal(|| false);
    let mut upload_progress = use_signal(|| 0usize);
    let navigator = use_navigator();

    // Determine current upload state
    let upload_state = if is_uploading() {
        if let Some(file_name) = uploaded_file() {
            let progress = (upload_progress() as f32 / 1_000_000.0 * 100.0).min(100.0);
            UploadState::Uploading {
                progress,
                file_name,
            }
        } else {
            UploadState::Uploading {
                progress: 0.0,
                file_name: "Unknown".to_string(),
            }
        }
    } else if let Some(file_name) = uploaded_file() {
        UploadState::Success { file_name }
    } else if is_dragging() {
        UploadState::Dragging
    } else {
        UploadState::Idle
    };

    let handle_drag_over = move |evt: DragEvent| {
        evt.prevent_default();
        is_dragging.set(true);
    };

    let handle_drag_leave = move |_evt: DragEvent| {
        is_dragging.set(false);
    };

    let handle_drop = move |evt: DragEvent| {
        evt.prevent_default();
        is_dragging.set(false);

        let drag_data = evt.data();
        let Some(engine) = drag_data.files() else {
            return;
        };
        let Some(file_name) = engine.files().into_iter().next() else {
            return;
        };

        // Get the file engine for upload
        let Some(file_engine) = drag_data.files() else {
            return;
        };

        // Start file upload
        is_uploading.set(true);
        uploaded_file.set(Some(file_name.clone()));
        upload_progress.set(0);

        let navigator = navigator;
        spawn(async move {
            match handle_file_upload_stream(file_engine, file_name, upload_progress).await {
                Ok(file_id) => {
                    is_uploading.set(false);
                    navigator.push(Route::Transcription { file: file_id });
                }
                Err(e) => {
                    error!("File upload failed: {}", e);
                    is_uploading.set(false);
                    uploaded_file.set(None);
                    upload_progress.set(0);
                }
            }
        });
    };

    let handle_file_input = move |evt: FormEvent| {
        let Some(engine) = evt.files() else {
            return;
        };
        let Some(file_name) = engine.files().into_iter().next() else {
            return;
        };

        // Get the file engine for upload
        let Some(file_engine) = evt.files() else {
            return;
        };

        // Start file upload
        is_uploading.set(true);
        uploaded_file.set(Some(file_name.clone()));
        upload_progress.set(0);

        let navigator = navigator;
        spawn(async move {
            match handle_file_upload_stream(file_engine, file_name, upload_progress).await {
                Ok(file_id) => {
                    is_uploading.set(false);
                    navigator.push(Route::Transcription { file: file_id });
                }
                Err(e) => {
                    error!("File upload failed: {}", e);
                    is_uploading.set(false);
                    uploaded_file.set(None);
                    upload_progress.set(0);
                }
            }
        });
    };

    let handle_upload_another = move |_evt: MouseEvent| {
        uploaded_file.set(None);
    };

    rsx! {
        UploadZone {
            state: upload_state,
            ondragover: handle_drag_over,
            ondragleave: handle_drag_leave,
            ondrop: handle_drop,
            onchange: handle_file_input,
            on_upload_another: handle_upload_another,
        }
    }
}

/// Handle file upload using the platform abstraction
async fn handle_file_upload_stream(
    file_engine: Arc<dyn FileEngine>,
    file_name: String,
    mut upload_progress: Signal<usize>,
) -> miette::Result<String> {
    use miette::miette;

    info!("Starting file processing: {}", file_name);

    // Read file content as bytes using Dioxus FileEngine
    let file_bytes = file_engine
        .read_file(&file_name)
        .await
        .ok_or_else(|| miette!("Failed to read file: {}", file_name))?;

    // Update progress
    let total_size = file_bytes.len();
    upload_progress.set(total_size / 2);

    // Use platform abstraction for file processing
    let platform = &*platform::PLATFORM;

    match platform
        .process_file(Bytes::from(file_bytes), file_name.clone())
        .await
    {
        Ok(file_id) => {
            upload_progress.set(total_size);
            info!("File processing completed, file ID: {}", file_id);
            Ok(file_id)
        }
        Err(e) => {
            error!("File processing failed: {}", e);
            Err(miette!("File processing failed: {}", e))
        }
    }
}
