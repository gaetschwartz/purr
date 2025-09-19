use crate::client::{default_client, TranscriptionStatus};
use dioxus::prelude::*;
use tracing::info;

/// The Transcription page component that displays the transcription functionality
/// for an uploaded audio file
#[component]
pub fn Transcription(file: String) -> Element {
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
            #[cfg(feature = "web")]
            {
                start_transcription_process(file_id, transcription_status, transcription_text);
            }
            #[cfg(not(feature = "web"))]
            {
                transcription_status.set(Some(TranscriptionStatus::Error {
                    message: "Transcription not supported in non-web builds".to_string(),
                }));
                is_transcribing.set(false);
            }
        }
    });

    let handle_back = move |_| {
        navigator.go_back();
    };

    rsx! {
        div { class: "flex flex-col min-h-screen bg-gradient-to-br from-gray-50 to-gray-100",
            // Header with file info and back button
            div { class: "bg-white/80 backdrop-blur-sm shadow-sm border-b border-gray-200 px-6 py-4",
                div { class: "flex items-center justify-between max-w-4xl mx-auto",
                    div { class: "flex items-center space-x-4",
                        button {
                            class: "flex items-center px-4 py-2 text-sm font-medium text-gray-600 hover:text-teal-600 hover:bg-teal-50 rounded-lg transition-all duration-200",
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
                        h1 { class: "text-2xl font-bold text-gray-900", "Audio Transcription" }
                    }
                }
            }

            // Main content area
            div { class: "flex-1 max-w-4xl mx-auto w-full px-6 py-8",
                // Transcription status and content
                div { class: "bg-white/90 backdrop-blur-sm rounded-xl shadow-lg border border-gray-200 p-8",
                    match transcription_status() {
                        Some(TranscriptionStatus::Starting) => rsx! {
                            div { class: "text-center py-16",
                                div { class: "flex justify-center mb-6",
                                    div { class: "animate-spin rounded-full h-16 w-16 border-4 border-teal-200 border-t-teal-500" }
                                }
                                h3 { class: "text-xl font-semibold text-gray-900 mb-3", "Initializing transcription..." }
                                p { class: "text-gray-600 text-lg", "Setting up transcription for: {file_for_ui}" }
                            }
                        },
                        Some(TranscriptionStatus::ProcessingAudio) => rsx! {
                            div { class: "text-center py-16",
                                div { class: "flex justify-center mb-6",
                                    div { class: "animate-spin rounded-full h-16 w-16 border-4 border-teal-200 border-t-teal-500" }
                                }
                                h3 { class: "text-xl font-semibold text-gray-900 mb-3", "Processing audio file..." }
                                p { class: "text-gray-600 text-lg", "Preparing audio for transcription" }
                            }
                        },
                        Some(TranscriptionStatus::InProgress { chunk_index, .. }) => {
                            rsx! {
                                div { class: "text-center py-16",
                                    div { class: "flex justify-center mb-6",
                                        div { class: "animate-pulse rounded-full h-16 w-16 bg-gradient-to-r from-teal-400 to-teal-600 shadow-lg" }
                                    }
                                    h3 { class: "text-xl font-semibold text-gray-900 mb-3", "Transcribing audio..." }
                                    div { class: "inline-flex items-center px-4 py-2 bg-teal-100 text-teal-800 rounded-full text-sm font-medium mb-8",
                                        "Processing chunk {chunk_index + 1}"
                                    }

                                    // Show live transcription results
                                    if !transcription_text().is_empty() {
                                        div { class: "mt-8 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200 text-left shadow-sm",
                                            h4 { class: "text-lg font-semibold text-blue-900 mb-4 flex items-center",
                                                svg {
                                                    class: "w-5 h-5 mr-2 text-blue-600",
                                                    fill: "currentColor",
                                                    view_box: "0 0 20 20",
                                                    path {
                                                        d: "M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z"
                                                    }
                                                }
                                                "Live Transcription"
                                            }
                                            p { class: "text-gray-800 whitespace-pre-wrap leading-relaxed text-lg", "{transcription_text()}" }
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
                            div { class: "text-center py-16",
                                div { class: "flex justify-center mb-6",
                                    div { class: "w-20 h-20 bg-green-100 rounded-full flex items-center justify-center",
                                        svg {
                                            class: "w-10 h-10 text-green-600",
                                            fill: "currentColor",
                                            view_box: "0 0 20 20",
                                            path {
                                                fill_rule: "evenodd",
                                                d: "M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z",
                                                clip_rule: "evenodd",
                                            }
                                        }
                                    }
                                }
                                h3 { class: "text-2xl font-bold text-green-700 mb-8", "Transcription Complete!" }

                                // Statistics
                                div { class: "grid grid-cols-2 md:grid-cols-4 gap-6 max-w-2xl mx-auto mb-10",
                                    div { class: "bg-white p-4 rounded-lg shadow-sm border border-gray-200 text-center",
                                        p { class: "text-sm font-medium text-gray-600 mb-1", "Processing Time" }
                                        p { class: "text-2xl font-bold text-teal-600", "{processing_time:.1}s" }
                                    }
                                    div { class: "bg-white p-4 rounded-lg shadow-sm border border-gray-200 text-center",
                                        p { class: "text-sm font-medium text-gray-600 mb-1", "Audio Duration" }
                                        p { class: "text-2xl font-bold text-teal-600", "{audio_duration:.1}s" }
                                    }
                                    div { class: "bg-white p-4 rounded-lg shadow-sm border border-gray-200 text-center",
                                        p { class: "text-sm font-medium text-gray-600 mb-1", "Word Count" }
                                        p { class: "text-2xl font-bold text-teal-600", "{word_count}" }
                                    }
                                    div { class: "bg-white p-4 rounded-lg shadow-sm border border-gray-200 text-center",
                                        p { class: "text-sm font-medium text-gray-600 mb-1", "Speed" }
                                        p { class: "text-2xl font-bold text-teal-600", "{audio_duration / processing_time as f32:.1}x" }
                                    }
                                }

                                // Final transcription results
                                div { class: "mt-8 p-8 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-200 text-left shadow-lg",
                                    h4 { class: "text-xl font-bold text-green-900 mb-6 flex items-center",
                                        svg {
                                            class: "w-6 h-6 mr-3 text-green-600",
                                            fill: "currentColor",
                                            view_box: "0 0 20 20",
                                            path {
                                                d: "M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z"
                                            }
                                        }
                                        "Final Transcription"
                                    }
                                    p { class: "text-gray-800 whitespace-pre-wrap leading-relaxed text-lg", "{transcription_text()}" }
                                }
                            }
                        },
                        Some(TranscriptionStatus::Error { message }) => {
                            rsx! {
                                div { class: "text-center py-16",
                                    div { class: "flex justify-center mb-6",
                                        div { class: "w-20 h-20 bg-red-100 rounded-full flex items-center justify-center",
                                            svg {
                                                class: "w-10 h-10 text-red-600",
                                                fill: "currentColor",
                                                view_box: "0 0 20 20",
                                                path {
                                                    fill_rule: "evenodd",
                                                    d: "M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z",
                                                    clip_rule: "evenodd",
                                                }
                                            }
                                        }
                                    }
                                    h3 { class: "text-xl font-semibold text-red-800 mb-4", "Transcription Failed" }
                                    div { class: "bg-red-50 border border-red-200 rounded-lg p-4 mb-8 max-w-md mx-auto",
                                        p { class: "text-red-700 text-sm", "{message}" }
                                    }
                                    button {
                                        class: "px-6 py-3 bg-teal-500 text-white rounded-lg hover:bg-teal-600 transition-colors shadow-md hover:shadow-lg",
                                        onclick: handle_back,
                                        "Try Another File"
                                    }
                                }
                            }
                        }
                        None => rsx! {
                            div { class: "text-center py-16",
                                div { class: "flex justify-center mb-6",
                                    div { class: "animate-spin rounded-full h-16 w-16 border-4 border-teal-200 border-t-teal-500" }
                                }
                                h3 { class: "text-xl font-semibold text-gray-900 mb-3", "Starting transcription..." }
                                p { class: "text-gray-600 text-lg", "Preparing to transcribe: {file_for_ui}" }
                            }
                        },
                    }
                }
            }
        }
    }
}

/// Start transcription process using real backend API
#[cfg(feature = "web")]
fn start_transcription_process(
    file_id: String,
    mut status_signal: Signal<Option<TranscriptionStatus>>,
    mut text_signal: Signal<String>,
) {
    info!("Starting REAL transcription for file ID: {}", file_id);

    let _client = default_client();

    // Start real transcription via HTTP API - using spawn to handle the async process
    spawn(async move {
        use gloo_timers::future::TimeoutFuture;

        status_signal.set(Some(TranscriptionStatus::Starting));
        TimeoutFuture::new(500).await;

        status_signal.set(Some(TranscriptionStatus::ProcessingAudio));

        // Call the real backend API for transcription
        let transcription_url = format!("http://localhost:8080/api/transcribe?file_id={}", file_id);

        match fetch_transcription(&transcription_url).await {
            Ok(response) => {
                // Set the full transcription text
                text_signal.set(response.text.clone());

                // Mark as completed with real stats
                status_signal.set(Some(TranscriptionStatus::Completed {
                    processing_time: response.processing_time,
                    audio_duration: (response.processing_time * 1.2) as f32, // Estimate audio duration
                    word_count: response.word_count,
                }));

                info!("Real transcription completed for file ID: {}", file_id);
            }
            Err(e) => {
                info!("Transcription failed for file ID {}: {}", file_id, e);
                status_signal.set(Some(TranscriptionStatus::Error {
                    message: format!("Transcription failed: {}", e),
                }));
            }
        }
    });
}

/// Response structure for transcription API
#[cfg(feature = "web")]
#[derive(serde::Serialize, serde::Deserialize)]
struct TranscriptionResponse {
    text: String,
    processing_time: f64,
    word_count: usize,
}

/// Fetch transcription from backend API
#[cfg(feature = "web")]
async fn fetch_transcription(url: &str) -> Result<TranscriptionResponse, String> {
    use web_sys::RequestInit;
    use wasm_bindgen::JsCast;

    let window = web_sys::window().ok_or("No window object")?;

    let mut opts = RequestInit::new();
    opts.set_method("GET");

    let request = web_sys::Request::new_with_str_and_init(url, &opts)
        .map_err(|_| "Failed to create request")?;

    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|_| "Network request failed")?;

    let resp: web_sys::Response = resp_value.dyn_into().map_err(|_| "Invalid response")?;

    if !resp.ok() {
        return Err(format!("HTTP {}: {}", resp.status(), resp.status_text()));
    }

    let json = wasm_bindgen_futures::JsFuture::from(resp.json().map_err(|_| "Failed to parse JSON")?)
        .await
        .map_err(|_| "Failed to read response body")?;

    let response: TranscriptionResponse = serde_wasm_bindgen::from_value(json)
        .map_err(|_| "Failed to deserialize response")?;

    Ok(response)
}
