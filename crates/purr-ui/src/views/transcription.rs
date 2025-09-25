use crate::{
    components::{Button, ButtonVariant, Card, IconType, TranscriptionDisplay},
    platform::{self, TranscriptionRequest, TranscriptionStatus},
};
use dioxus::prelude::*;
use futures::StreamExt;
use purr_common::platform::{FileId, FileSource};
use tracing::{error, info};

/// The Transcription page component that displays the transcription functionality
/// for an uploaded audio file
#[component]
pub fn Transcription(file: FileId) -> Element {
    let navigator = use_navigator();
    let transcription_status = use_signal(|| None::<TranscriptionStatus>);
    let transcription_text = use_signal(String::new);
    let mut is_transcribing = use_signal(|| false);

    // Start transcription when component mounts
    let file_for_ui = file.clone();
    use_effect(move || {
        let file_id = file.clone();
        if !is_transcribing() {
            is_transcribing.set(true);
            start_transcription_process(file_id, transcription_status, transcription_text);
        }
    });

    let handle_back = move |_| {
        navigator.go_back();
    };

    rsx! {
        div { class: "main-layout",
            // Header with file info and back button
            div { class: "backdrop-glass shadow-sm border-b border-gray-200 px-6 py-4",
                div { class: "flex items-center justify-between max-w-4xl mx-auto",
                    div { class: "flex items-center space-x-4",
                        Button {
                            variant: ButtonVariant::Ghost,
                            icon: Some(IconType::BackArrow),
                            onclick: handle_back,
                            "Back"
                        }
                        h1 { class: "text-2xl font-bold text-gray-900", "Audio Transcription" }
                    }
                }
            }

            // Main content area
            div { class: "content-container",
                // Transcription status and content
                Card {
                    TranscriptionDisplay {
                        status: transcription_status(),
                        text: transcription_text(),
                        file_name: file_for_ui.clone(),
                    }

                    // Add back button for error state
                    if let Some(TranscriptionStatus::Error { .. }) = transcription_status() {
                        div { class: "mt-8 text-center",
                            Button {
                                variant: ButtonVariant::Primary,
                                onclick: handle_back,
                                "Try Another File"
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Start transcription process using platform abstraction
fn start_transcription_process(
    file_id: FileId,
    mut status_signal: Signal<Option<TranscriptionStatus>>,
    mut text_signal: Signal<String>,
) {
    info!("Starting transcription for file ID: {}", file_id);

    spawn(async move {
        // Get platform implementation
        let platform = match platform::get_platform().await {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to get platform: {}", e);
                status_signal.set(Some(TranscriptionStatus::InitFailed {
                    component: "platform".to_string(),
                    reason: e.to_string(),
                    error_details: None,
                }));
                return;
            }
        };

        let request = TranscriptionRequest {
            language: None,
            translate: false,
            file: FileSource::Uploaded(file_id.clone()),
        };

        // Start transcription with streaming updates
        match platform.transcribe(request).await {
            Ok(mut stream) => {
                let mut full_text = String::new();

                // Process streaming updates
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(status) => {
                            // Update status signal
                            status_signal.set(Some(status.clone()));

                            // Accumulate text for in-progress updates
                            if let TranscriptionStatus::InProgress { text, .. } = &status {
                                full_text.push_str(text);
                                full_text.push(' ');
                                text_signal.set(full_text.clone());
                            }

                            // Check if completed or errored
                            match status {
                                TranscriptionStatus::Completed { .. } => {
                                    info!("Transcription completed for file ID: {}", file_id);
                                    break;
                                }
                                TranscriptionStatus::Error {
                                    context,
                                    error_message,
                                } => {
                                    error!("Transcription error: {}: {}", context, error_message);
                                    break;
                                }
                                _ => {}
                            }
                        }
                        Err(e) => {
                            error!("Stream error: {}", e);
                            status_signal.set(Some(TranscriptionStatus::Error {
                                context: "stream".to_string(),
                                error_message: format!("Stream error: {e}"),
                            }));
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                error!("Failed to start transcription: {}", e);
                status_signal.set(Some(TranscriptionStatus::Error {
                    context: "transcription_start".to_string(),
                    error_message: format!("Failed to start transcription: {e}"),
                }));
            }
        }
    });
}
