use crate::{
    server::fs::{upload_file, UploadStatus},
    utils::BytesExt,
    Route,
};
use bytes::Bytes;
use dioxus::{
    html::{FileEngine, HasFileData},
    prelude::*,
};
use futures::{channel::mpsc, SinkExt, StreamExt};
use miette::miette;
use std::sync::Arc;
use tracing::{error, info};

#[component]
pub fn DragDropZone() -> Element {
    let mut is_dragging = use_signal(|| false);
    let mut uploaded_file = use_signal(|| None::<String>);
    let mut is_uploading = use_signal(|| false);
    let mut upload_progress = use_signal(|| 0usize);
    let navigator = use_navigator();

    let handle_drag_over = move |evt: Event<DragData>| {
        evt.prevent_default();
        is_dragging.set(true);
    };

    let handle_drag_leave = move |_evt: Event<DragData>| {
        is_dragging.set(false);
    };

    let handle_drop = move |evt: Event<DragData>| {
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

    let handle_file_input = move |evt: Event<FormData>| {
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

    let border_class = if is_dragging() {
        "border-4 border-dashed border-teal-500 bg-teal-50"
    } else {
        "border-4 border-dashed border-gray-300 hover:border-gray-400"
    };

    rsx! {
      div { class: "flex items-center justify-center w-full h-full",
        div {
          class: "flex flex-col items-center justify-center w-full h-full {border_class} rounded-lg cursor-pointer transition-colors duration-200",
          ondragover: handle_drag_over,
          ondragleave: handle_drag_leave,
          ondrop: handle_drop,

          // Hidden file input for click-to-browse functionality
          input {
            r#type: "file",
            class: "hidden",
            id: "file-input",
            name: "audio-file",
            multiple: false,
            // Accept audio types - includes both MIME types and file extensions
            accept: "audio/mpeg, audio/wav, audio/x-wav, audio/wave, audio/aac, audio/flac, audio/ogg, audio/webm, audio/mp4, audio/x-m4a, .mp3, .wav, .aac, .flac, .ogg, .webm, .m4a",
            onchange: handle_file_input,
          }

          // Click handler that triggers the file input
          label {
            r#for: "file-input",
            class: "flex flex-col items-center justify-center w-full h-full cursor-pointer",

            if is_uploading() {
              div { class: "flex flex-col items-center",
                div { class: "animate-spin rounded-full h-8 w-8 border-b-2 border-teal-600 mb-4" }
                p { class: "text-gray-600", "Uploading file..." }
                if upload_progress() > 0 {
                  p { class: "text-sm text-gray-500",
                    "{upload_progress()} bytes uploaded"
                  }
                }
              }
            } else if let Some(file_name) = uploaded_file() {
              div { class: "flex flex-col items-center",
                svg {
                  class: "w-16 h-16 max-w-16 max-h-16 mb-4 text-green-500",
                  fill: "currentColor",
                  view_box: "0 0 20 20",
                  path {
                    fill_rule: "evenodd",
                    d: "M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z",
                    clip_rule: "evenodd",
                  }
                }
                p { class: "text-green-600 font-medium",
                  "File uploaded successfully!"
                }
                p { class: "text-gray-500 text-sm break-all break-words max-w-full",
                  "{file_name}"
                }
                button {
                  class: "mt-2 px-4 py-2 bg-teal-500 text-white rounded hover:bg-teal-600 transition-colors",
                  onclick: move |evt| {
                      evt.prevent_default();
                      evt.stop_propagation();
                      uploaded_file.set(None);
                  },
                  "Upload Another File"
                }
              }
            } else {
              div { class: "flex flex-col items-center",
                svg {
                  class: "w-8 h-8 mb-4 text-gray-400",
                  fill: "none",
                  stroke: "currentColor",
                  view_box: "0 0 48 48",
                  path {
                    stroke_linecap: "round",
                    stroke_linejoin: "round",
                    stroke_width: "2",
                    d: "M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02",
                  }
                }
                p { class: "mb-2 text-sm text-gray-500",
                  span { class: "font-semibold", "Click to upload" }
                  " or drag and drop"
                }
                p { class: "text-xs text-gray-500", "Any file type supported" }
              }
            }
          }
        }
      }
    }
}

/// Handle file upload using streaming WebSocket upload
async fn handle_file_upload_stream(
    file_engine: Arc<dyn FileEngine>,
    file_name: String,
    mut upload_progress: Signal<usize>,
) -> miette::Result<String> {
    info!("Starting streaming file upload: {}", file_name);

    // Read file content as bytes using Dioxus FileEngine
    let file_bytes = file_engine
        .read_file(&file_name)
        .await
        .ok_or_else(|| miette!("Failed to read file: {}", file_name))?;

    // Create a stream from the file bytes
    let (mut tx, rx) = mpsc::channel::<Result<Bytes, String>>(100);

    // Send file data in chunks
    let chunk_size = 8192; // 8KB chunks
    let total_size = file_bytes.len();

    spawn(async move {
        let file_bytes = Bytes::from(file_bytes);
        for (i, chunk) in file_bytes.chunks(chunk_size).enumerate() {
            if tx.send(Ok(chunk)).await.is_err() {
                error!("Failed to send chunk {}", i);
                break;
            }

            // Small delay to prevent overwhelming the connection
            gloo_timers::future::TimeoutFuture::new(10).await;
        }

        // Close the sender to indicate end of stream
        tx.close().await.unwrap();
    });

    // Convert the receiver to a BoxedStream
    let byte_stream = rx.map(|result| result.map_err(ServerFnError::new)).boxed();

    // Call the upload server function
    match upload_file(byte_stream.into()).await {
        Ok(mut status_stream) => {
            let mut final_file_id = None;

            // Process status updates
            while let Some(status_result) = status_stream.next().await {
                match status_result {
                    Ok(UploadStatus::InProgress { bytes_received }) => {
                        info!("Upload progress: {} / {} bytes", bytes_received, total_size);
                        upload_progress.set(bytes_received);
                    }
                    Ok(UploadStatus::Completed {
                        total_bytes,
                        file_id,
                    }) => {
                        info!(
                            "Upload completed: {} bytes, file ID: {}",
                            total_bytes, file_id
                        );
                        final_file_id = Some(file_id);
                        break;
                    }
                    Err(e) => {
                        error!("Upload error: {}", e);
                        return Err(miette!("Upload failed: {}", e));
                    }
                }
            }

            final_file_id.ok_or_else(|| miette!("Upload completed but no file ID received"))
        }
        Err(e) => {
            error!("Failed to start upload: {}", e);
            Err(miette!("Failed to start upload: {}", e))
        }
    }
}
