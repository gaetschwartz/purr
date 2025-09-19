use crate::{
    client::default_client,
    Route,
};
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
        "border-2 border-dashed border-teal-500 bg-teal-50/50"
    } else {
        "border-2 border-dashed border-gray-300 hover:border-teal-400 hover:bg-gray-50/50"
    };

    rsx! {
      div { class: "flex items-center justify-center w-full h-full min-h-screen bg-gradient-to-br from-gray-50 to-gray-100",
        div {
          class: "flex flex-col items-center justify-center w-full max-w-lg mx-auto h-96 {border_class} rounded-xl cursor-pointer transition-all duration-300 ease-in-out transform hover:scale-105 bg-white shadow-lg",
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
            accept: "audio/mpeg, audio/wav, audio/x-wav, audio/wave, audio/aac, audio/flac, audio/ogg, audio/webm, audio/mp4, audio/x-m4a, .mp3, .wav, .aac, .flac, .ogg, .webm, .m4a, .mp4",
            onchange: handle_file_input,
          }

          // Click handler that triggers the file input
          label {
            r#for: "file-input",
            class: "flex flex-col items-center justify-center w-full h-full cursor-pointer p-8",

            if is_uploading() {
              div { class: "flex flex-col items-center text-center",
                div { class: "animate-spin rounded-full h-12 w-12 border-b-2 border-teal-500 mb-6" }
                h3 { class: "text-lg font-medium text-gray-900 mb-2", "Uploading file..." }
                if upload_progress() > 0 {
                  div { class: "w-full max-w-xs mx-auto",
                    div { class: "bg-gray-200 rounded-full h-2 mb-2",
                      div {
                        class: "bg-teal-500 h-2 rounded-full transition-all duration-300",
                        style: "width: {(upload_progress() as f32 / 1_000_000.0 * 100.0).min(100.0)}%"
                      }
                    }
                    p { class: "text-sm text-gray-600",
                      "{upload_progress() / 1000} KB uploaded"
                    }
                  }
                }
              }
            } else if let Some(file_name) = uploaded_file() {
              div { class: "flex flex-col items-center text-center",
                div { class: "w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-4",
                  svg {
                    class: "w-8 h-8 text-green-600",
                    fill: "currentColor",
                    view_box: "0 0 20 20",
                    path {
                      fill_rule: "evenodd",
                      d: "M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z",
                      clip_rule: "evenodd",
                    }
                  }
                }
                h3 { class: "text-lg font-medium text-green-700 mb-2",
                  "File uploaded successfully!"
                }
                p { class: "text-gray-600 text-sm break-all break-words max-w-full mb-4",
                  "{file_name}"
                }
                button {
                  class: "px-6 py-2 bg-teal-500 text-white rounded-lg hover:bg-teal-600 transition-colors shadow-md hover:shadow-lg",
                  onclick: move |evt| {
                      evt.prevent_default();
                      evt.stop_propagation();
                      uploaded_file.set(None);
                  },
                  "Upload Another File"
                }
              }
            } else {
              div { class: "flex flex-col items-center text-center",
                div { class: "w-16 h-16 bg-teal-100 rounded-full flex items-center justify-center mb-6",
                  svg {
                    class: "w-8 h-8 text-teal-600",
                    fill: "none",
                    stroke: "currentColor",
                    view_box: "0 0 24 24",
                    path {
                      stroke_linecap: "round",
                      stroke_linejoin: "round",
                      stroke_width: "1.5",
                      d: "M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10",
                    }
                  }
                }
                h2 { class: "text-xl font-semibold text-gray-900 mb-2", "Upload Audio File" }
                p { class: "text-gray-600 mb-1",
                  span { class: "font-medium", "Click to upload" }
                  " or drag and drop"
                }
                p { class: "text-sm text-gray-500", "Audio files supported (MP3, WAV, AAC, FLAC, OGG, M4A)" }
              }
            }
          }
        }
      }
    }
}

/// Handle file upload using HTTP API
#[cfg(feature = "web")]
async fn handle_file_upload_stream(
    file_engine: Arc<dyn FileEngine>,
    file_name: String,
    mut upload_progress: Signal<usize>,
) -> miette::Result<String> {
    use miette::miette;

    info!("Starting file upload: {}", file_name);

    // Read file content as bytes using Dioxus FileEngine
    let file_bytes = file_engine
        .read_file(&file_name)
        .await
        .ok_or_else(|| miette!("Failed to read file: {}", file_name))?;

    let client = default_client();

    // Simulate progress updates
    let total_size = file_bytes.len();
    upload_progress.set(total_size / 2);

    // Upload file via HTTP
    match client.upload_file(file_name.clone(), file_bytes).await {
        Ok(file_id) => {
            upload_progress.set(total_size);
            info!("Upload completed, file ID: {}", file_id);
            Ok(file_id)
        }
        Err(e) => {
            error!("Upload failed: {}", e);
            Err(miette!("Upload failed: {}", e))
        }
    }
}

/// Placeholder for non-web builds
#[cfg(not(feature = "web"))]
async fn handle_file_upload_stream(
    _file_engine: Arc<dyn FileEngine>,
    _file_name: String,
    _upload_progress: Signal<usize>,
) -> miette::Result<String> {
    use miette::miette;
    Err(miette!("File upload not supported in non-web builds"))
}
