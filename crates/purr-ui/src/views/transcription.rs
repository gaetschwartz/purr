use crate::{
    client::{default_client, TranscriptionStatus},
    components::{Button, ButtonVariant, Card, IconType, TranscriptionDisplay},
};
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
        div { class: "main-layout",
            // Header with file info and back button
            div { class: "backdrop-glass shadow-sm border-b border-gray-200 px-6 py-4",
                div { class: "flex items-center justify-between max-w-4xl mx-auto",
                    div { class: "flex items-center space-x-4",
                        Button {
                            variant: ButtonVariant::Ghost,
                            icon: Some(IconType::BackArrow),
                            onclick: move |evt| handle_back(evt),
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
                                onclick: move |evt| handle_back(evt),
                                "Try Another File"
                            }
                        }
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

    let opts = RequestInit::new();
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
