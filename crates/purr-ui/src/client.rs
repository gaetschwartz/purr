/// Client-side API for communicating with the backend server
/// This module provides HTTP and WebSocket communication for WASM builds

use serde::{Deserialize, Serialize};

/// Upload status for file uploads
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum UploadStatus {
    InProgress { bytes_received: usize },
    Completed { file_id: String },
    Error { message: String },
}

/// Transcription status updates
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum TranscriptionStatus {
    Starting,
    ProcessingAudio,
    InProgress {
        chunk_index: usize,
        text: String,
        start_time: f64,
        end_time: f64,
    },
    Completed {
        processing_time: f64,
        audio_duration: f32,
        word_count: usize,
    },
    Error { message: String },
}

/// Client API for backend communication
pub struct ApiClient {
    base_url: String,
}

impl ApiClient {
    pub fn new(base_url: String) -> Self {
        Self { base_url }
    }

    /// Upload a file to the backend via HTTP POST
    #[cfg(feature = "web")]
    pub async fn upload_file(&self, file_name: String, file_data: Vec<u8>) -> Result<String, String> {
        use web_sys::{window, FormData, Request, RequestInit, Response};
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;

        tracing::info!("Uploading file: {} ({} bytes)", file_name, file_data.len());

        let window = window().ok_or("No window available")?;

        // Create FormData for file upload
        let form_data = FormData::new().map_err(|_| "Failed to create FormData")?;

        // Convert file data to Uint8Array
        let uint8_array = js_sys::Uint8Array::new_with_length(file_data.len() as u32);
        uint8_array.copy_from(&file_data);

        // Create a Blob from the data
        let blob_parts = js_sys::Array::new();
        blob_parts.push(&uint8_array);

        let blob_property_bag = web_sys::BlobPropertyBag::new();
        blob_property_bag.set_type("application/octet-stream");

        let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(
            &blob_parts,
            &blob_property_bag,
        ).map_err(|_| "Failed to create blob")?;

        // Add file to form data
        form_data
            .append_with_blob_and_filename("file", &blob, &file_name)
            .map_err(|_| "Failed to append file to FormData")?;

        // Create request
        let opts = RequestInit::new();
        opts.set_method("POST");
        opts.set_body(form_data.as_ref());

        let url = format!("{}/api/upload", self.base_url);
        let request = Request::new_with_str_and_init(&url, &opts)
            .map_err(|_| "Failed to create request")?;

        // Send request
        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|_| "Failed to fetch")?;

        let resp: Response = resp_value
            .dyn_into()
            .map_err(|_| "Failed to cast to Response")?;

        if !resp.ok() {
            return Err(format!("Upload failed with status: {}", resp.status()));
        }

        // Parse response
        let json = JsFuture::from(resp.json().map_err(|_| "Failed to get JSON")?)
            .await
            .map_err(|_| "Failed to parse JSON")?;

        #[cfg(feature = "web")]
        let response: serde_json::Value = serde_wasm_bindgen::from_value(json)
            .map_err(|_| "Failed to deserialize response")?;

        response
            .get("file_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "No file_id in response".to_string())
    }

    /// Get transcription via HTTP GET
    #[cfg(feature = "web")]
    pub async fn get_transcription(
        &self,
        file_id: String,
        language: Option<String>,
        translate: bool,
    ) -> Result<String, String> {
        use web_sys::{window, Request, RequestInit, Response};
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;

        tracing::info!("Requesting transcription for file: {}", file_id);

        let window = window().ok_or("No window available")?;

        // Build query parameters
        let mut query_params = vec![
            format!("file_id={}", file_id),
            format!("translate={}", translate),
        ];
        if let Some(lang) = language {
            query_params.push(format!("language={}", lang));
        }

        let url = format!("{}/api/transcribe?{}", self.base_url, query_params.join("&"));

        // Create request
        let opts = RequestInit::new();
        opts.set_method("GET");

        let request = Request::new_with_str_and_init(&url, &opts)
            .map_err(|_| "Failed to create request")?;

        // Send request
        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|_| "Failed to fetch")?;

        let resp: Response = resp_value
            .dyn_into()
            .map_err(|_| "Failed to cast to Response")?;

        if !resp.ok() {
            return Err(format!("Transcription failed with status: {}", resp.status()));
        }

        // Parse response
        let json = JsFuture::from(resp.json().map_err(|_| "Failed to get JSON")?)
            .await
            .map_err(|_| "Failed to parse JSON")?;

        #[cfg(feature = "web")]
        let response: serde_json::Value = serde_wasm_bindgen::from_value(json)
            .map_err(|_| "Failed to deserialize response")?;

        response
            .get("text")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "No text in response".to_string())
    }

    /// Start transcription via WebSocket (simulated with HTTP for now)
    #[cfg(feature = "web")]
    pub async fn start_transcription_ws(
        &self,
        file_id: String,
        language: Option<String>,
        translate: bool,
        status_callback: impl Fn(TranscriptionStatus) + 'static,
    ) -> Result<(), String> {
        use gloo_timers::future::TimeoutFuture;

        tracing::info!("Starting transcription for file: {}", file_id);

        // Simulate transcription progress
        status_callback(TranscriptionStatus::Starting);
        TimeoutFuture::new(500).await;

        status_callback(TranscriptionStatus::ProcessingAudio);
        TimeoutFuture::new(1000).await;

        // Get the actual transcription from the backend
        match self.get_transcription(file_id.clone(), language, translate).await {
            Ok(text) => {
                // Simulate progress chunks by splitting the text
                let words: Vec<&str> = text.split_whitespace().collect();
                let chunk_size = words.len().max(1) / 2; // Split into 2 chunks

                if !words.is_empty() {
                    // First chunk
                    let first_chunk: Vec<&str> = words.iter().take(chunk_size).cloned().collect();
                    let first_text = first_chunk.join(" ");

                    status_callback(TranscriptionStatus::InProgress {
                        chunk_index: 0,
                        text: first_text,
                        start_time: 0.0,
                        end_time: 2.5,
                    });
                    TimeoutFuture::new(1000).await;

                    // Second chunk
                    let second_chunk: Vec<&str> = words.iter().skip(chunk_size).cloned().collect();
                    let second_text = second_chunk.join(" ");

                    if !second_text.is_empty() {
                        status_callback(TranscriptionStatus::InProgress {
                            chunk_index: 1,
                            text: format!(" {}", second_text),
                            start_time: 2.5,
                            end_time: 5.0,
                        });
                        TimeoutFuture::new(1000).await;
                    }
                }

                // Completion
                status_callback(TranscriptionStatus::Completed {
                    processing_time: 3.5,
                    audio_duration: 5.0,
                    word_count: words.len(),
                });

                Ok(())
            }
            Err(e) => {
                status_callback(TranscriptionStatus::Error { message: e.clone() });
                Err(e)
            }
        }
    }
}

/// Default API client for the current environment
#[cfg(feature = "web")]
pub fn default_client() -> ApiClient {
    // In WASM, assume backend is on port 8080
    ApiClient::new("http://localhost:8080".to_string())
}

#[cfg(not(feature = "web"))]
pub fn default_client() -> ApiClient {
    // For non-web builds, this shouldn't be used
    ApiClient::new("http://localhost:8080".to_string())
}