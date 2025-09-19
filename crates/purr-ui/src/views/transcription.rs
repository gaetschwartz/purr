use crate::server::whisper::{start_transcription, TranscriptionRequest, TranscriptionStatus};
use dioxus::prelude::*;
use futures::{stream, StreamExt as _};
use tracing::{error, info};

/// The Transcription page component that displays the transcription functionality
/// for an uploaded audio file
#[component]
pub fn Transcription(file: String) -> Element {
    let navigator = use_navigator();
    let mut transcription_status = use_signal(|| None::<TranscriptionStatus>);
    let transcription_text = use_signal(String::new);
    let mut is_transcribing = use_signal(|| false);

    // Start transcription when component mounts
    let file_for_ui = file.clone();
    use_effect(move || {
        let file_id = file.clone();
        if !is_transcribing() {
            is_transcribing.set(true);
            spawn(async move {
                if let Err(e) =
                    start_transcription_process(file_id, transcription_status, transcription_text)
                        .await
                {
                    error!("Transcription failed: {}", e);
                    transcription_status.set(Some(TranscriptionStatus::Error {
                        message: format!("Failed to start transcription: {e}"),
                    }));
                    is_transcribing.set(false);
                }
            });
        }
    });

    let handle_back = move |_| {
        navigator.go_back();
    };

    rsx! {
        div { class: "flex flex-col h-screen bg-gray-50",
            // Header with file info and back button
            div { class: "bg-white shadow-sm border-b border-gray-200 px-6 py-4",
                div { class: "flex items-center justify-between max-w-4xl mx-auto",
                    div { class: "flex items-center space-x-4",
                        button {
                            class: "flex items-center px-3 py-2 text-sm font-medium text-gray-600 hover:text-teal-600 transition-colors",
                            onclick: handle_back,
                            svg {
                                class: "w-4 h-4 mr-2",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M15 19l-7-7 7-7",
                                }
                            }
                            "Back"
                        }
                        h1 { class: "text-xl font-semibold text-gray-900", "Transcription" }
                    }
                }
            }

            // Main content area
            div { class: "flex-1 max-w-4xl mx-auto w-full px-6 py-8",
                // Transcription status and content
                div { class: "bg-white rounded-lg shadow-sm border border-gray-200 p-6",
                    match transcription_status() {
                        Some(TranscriptionStatus::Starting) => rsx! {
                            div { class: "text-center py-12",
                                div { class: "flex justify-center mb-4",
                                    div { class: "animate-spin rounded-full h-12 w-12 border-b-2 border-teal-500" }
                                }
                                h3 { class: "text-lg font-medium text-gray-900 mb-2", "Initializing transcription..." }
                                p { class: "text-gray-600", "Setting up transcription for: {file_for_ui}" }
                            }
                        },
                        Some(TranscriptionStatus::ProcessingAudio) => rsx! {
                            div { class: "text-center py-12",
                                div { class: "flex justify-center mb-4",
                                    div { class: "animate-spin rounded-full h-12 w-12 border-b-2 border-teal-500" }
                                }
                                h3 { class: "text-lg font-medium text-gray-900 mb-2", "Processing audio file..." }
                                p { class: "text-gray-600", "Preparing audio for transcription" }
                            }
                        },
                        Some(TranscriptionStatus::InProgress { chunk_index, .. }) => {
                            rsx! {
                                div { class: "text-center py-12",
                                    div { class: "flex justify-center mb-4",
                                        div { class: "animate-pulse rounded-full h-12 w-12 bg-teal-500" }
                                    }
                                    h3 { class: "text-lg font-medium text-gray-900 mb-2", "Transcribing audio..." }
                                    p { class: "text-gray-600 mb-6", "Processing chunk {chunk_index + 1}..." }
                                
                                    // Show live transcription results
                                    if !transcription_text().is_empty() {
                                        div { class: "mt-8 p-4 bg-white rounded-lg border border-gray-200 text-left",
                                            h4 { class: "text-sm font-medium text-gray-700 mb-2", "Live Transcription:" }
                                            p { class: "text-gray-900 whitespace-pre-wrap", "{transcription_text()}" }
                                        }
                                    }
                                }
                            }
                        }
                        Some(
                            TranscriptionStatus::Completed {
                                processing_time,
                                audio_duration,
                                word_count,
                            },
                        ) => rsx! {
                            div { class: "text-center py-12",
                                div { class: "flex justify-center mb-4",
                                    svg {
                                        class: "w-12 h-12 text-green-500",
                                        fill: "currentColor",
                                        view_box: "0 0 20 20",
                                        path {
                                            fill_rule: "evenodd",
                                            d: "M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z",
                                            clip_rule: "evenodd",
                                        }
                                    }
                                }
                                h3 { class: "text-lg font-medium text-gray-900 mb-2", "Transcription Complete!" }
                            
                                // Statistics
                                div { class: "grid grid-cols-2 gap-4 max-w-md mx-auto mb-6 text-sm text-gray-600",
                                    div { class: "text-center",
                                        p { class: "font-medium", "Processing Time" }
                                        p { "{processing_time:.1}s" }
                                    }
                                    div { class: "text-center",
                                        p { class: "font-medium", "Audio Duration" }
                                        p { "{audio_duration:.1}s" }
                                    }
                                    div { class: "text-center",
                                        p { class: "font-medium", "Word Count" }
                                        p { "{word_count}" }
                                    }
                                    div { class: "text-center",
                                        p { class: "font-medium", "Speed" }
                                        p { "{audio_duration / processing_time as f32:.1}x" }
                                    }
                                }
                            
                                // Final transcription results
                                div { class: "mt-8 p-6 bg-white rounded-lg border border-gray-200 text-left",
                                    h4 { class: "text-lg font-medium text-gray-900 mb-4", "Final Transcription:" }
                                    p { class: "text-gray-900 whitespace-pre-wrap leading-relaxed", "{transcription_text()}" }
                                }
                            }
                        },
                        Some(TranscriptionStatus::Error { message }) => {
                            rsx! {
                                div { class: "text-center py-12",
                                    div { class: "flex justify-center mb-4",
                                        svg {
                                            class: "w-12 h-12 text-red-500",
                                            fill: "currentColor",
                                            view_box: "0 0 20 20",
                                            path {
                                                fill_rule: "evenodd",
                                                d: "M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z",
                                                clip_rule: "evenodd",
                                            }
                                        }
                                    }
                                    h3 { class: "text-lg font-medium text-red-900 mb-2", "Transcription Failed" }
                                    p { class: "text-red-600 mb-6", "{message}" }
                                    button {
                                        class: "px-4 py-2 bg-teal-500 text-white rounded hover:bg-teal-600 transition-colors",
                                        onclick: handle_back,
                                        "Try Another File"
                                    }
                                }
                            }
                        }
                        None => rsx! {
                            div { class: "text-center py-12",
                                div { class: "flex justify-center mb-4",
                                    div { class: "animate-spin rounded-full h-12 w-12 border-b-2 border-teal-500" }
                                }
                                h3 { class: "text-lg font-medium text-gray-900 mb-2", "Starting transcription..." }
                                p { class: "text-gray-600", "Preparing to transcribe: {file_for_ui}" }
                            }
                        },
                    }
                }
            }
        }
    }
}

/// Start transcription process with streaming
async fn start_transcription_process(
    file_id: String,
    mut status_signal: Signal<Option<TranscriptionStatus>>,
    mut text_signal: Signal<String>,
) -> Result<(), String> {
    info!("Starting transcription for file ID: {}", file_id);

    // Create transcription request
    let request = TranscriptionRequest {
        file_id: file_id.clone(),
        language: None, // Auto-detect
        translate: false,
    };

    // Create a stream containing just the request
    let request_stream = stream::once(async { Ok(request) }).boxed();

    // Call the streaming server function
    match start_transcription(request_stream.into()).await {
        Ok(mut status_stream) => {
            info!("Connected to transcription stream for file ID: {}", file_id);

            // Process status updates from the stream
            while let Some(status_result) = status_stream.next().await {
                match status_result {
                    Ok(status) => {
                        let is_completed = matches!(status, TranscriptionStatus::Completed { .. });
                        let is_error = matches!(status, TranscriptionStatus::Error { .. });

                        match &status {
                            TranscriptionStatus::Starting => {
                                info!("Transcription starting...");
                            }
                            TranscriptionStatus::ProcessingAudio => {
                                info!("Processing audio...");
                            }
                            TranscriptionStatus::InProgress {
                                chunk_index,
                                text,
                                start_time,
                                end_time,
                            } => {
                                info!(
                                    "Transcription chunk {}: {} ({:.1}s - {:.1}s)",
                                    chunk_index, text, start_time, end_time
                                );

                                // Append the new text chunk
                                text_signal.with_mut(|current_text| {
                                    if !current_text.is_empty() && !text.starts_with(' ') {
                                        current_text.push(' ');
                                    }
                                    current_text.push_str(text);
                                });
                            }
                            TranscriptionStatus::Completed {
                                processing_time,
                                audio_duration,
                                word_count,
                            } => {
                                info!(
                                    "Transcription completed in {:.2}s: {} words from {:.1}s audio",
                                    processing_time, word_count, audio_duration
                                );
                            }
                            TranscriptionStatus::Error { message } => {
                                error!("Transcription error: {}", message);
                                let error_msg = message.clone();
                                status_signal.set(Some(status));
                                return Err(error_msg);
                            }
                        }

                        // Set the status after the match (except for error case handled above)
                        if !is_error {
                            status_signal.set(Some(status));
                        }

                        // Break on completion
                        if is_completed {
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Stream error during transcription: {}", e);
                        let error_status = TranscriptionStatus::Error {
                            message: format!("Stream error: {e}"),
                        };
                        status_signal.set(Some(error_status));
                        return Err(e.to_string());
                    }
                }
            }

            info!("Transcription stream completed for file ID: {}", file_id);
            Ok(())
        }
        Err(e) => {
            error!("Failed to start transcription stream: {}", e);
            status_signal.set(Some(TranscriptionStatus::Error {
                message: format!("Failed to start transcription: {e}"),
            }));
            Err(e.to_string())
        }
    }
}
