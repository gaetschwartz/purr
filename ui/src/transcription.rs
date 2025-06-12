use dioxus::prelude::*;
use std::path::PathBuf;

// Server-side imports
#[cfg(feature = "server")]
use purr_core::{transcribe_audio_file, TranscriptionConfig};

// Client-side imports (when api crate is available)
#[cfg(all(feature = "api", not(feature = "server")))]
use api::transcribe_audio;

#[derive(Clone, Debug)]
pub struct TranscriptionState {
    pub is_transcribing: bool,
    pub result: Option<String>,
    pub error: Option<String>,
    pub file_path: Option<PathBuf>,
}

impl Default for TranscriptionState {
    fn default() -> Self {
        Self {
            is_transcribing: false,
            result: None,
            error: None,
            file_path: None,
        }
    }
}

#[component]
pub fn TranscriptionView() -> Element {
    let state = use_signal(TranscriptionState::default);

    // File picker - platform-specific implementation
    let pick_file = move |_| {
        let mut state = state.clone();
        spawn(async move {
            if let Some(path) = open_file_dialog().await {
                state.write().file_path = Some(path.clone());
                transcribe_file(path, state).await;
            }
        });
    };

    // Placeholder for drag and drop handlers (not implemented yet for web)
    let on_drop = move |_evt: DragEvent| {
        // TODO: Implement drag and drop for web platform
    };

    let on_drag_over = move |evt: DragEvent| {
        evt.prevent_default();
    };

    let on_drag_enter = move |evt: DragEvent| {
        evt.prevent_default();
    };

    let state_read = state.read();

    rsx! {
        div {
            class: "transcription-container",
            style: "padding: 2rem; max-width: 800px; margin: 0 auto;",
            
            h1 {
                style: "text-align: center; margin-bottom: 2rem; color: #333;",
                "🎵 Audio Transcription"
            }

            // Drag and drop zone
            div {
                class: "drop-zone",
                style: "
                    border: 3px dashed #ccc;
                    border-radius: 10px;
                    padding: 3rem;
                    text-align: center;
                    margin-bottom: 2rem;
                    background: #f9f9f9;
                    transition: all 0.3s ease;
                    cursor: pointer;
                ",
                ondrop: on_drop,
                ondragover: on_drag_over,
                ondragenter: on_drag_enter,
                onclick: pick_file,

                if state_read.is_transcribing {
                    div {
                        style: "color: #666;",
                        "🎙️ Transcribing audio..."
                        div {
                            style: "margin-top: 1rem;",
                            "Please wait while we process your audio file."
                        }
                    }
                } else {
                    div {
                        div {
                            style: "font-size: 3rem; margin-bottom: 1rem;",
                            "📁"
                        }
                        h3 {
                            style: "margin-bottom: 1rem; color: #555;",
                            "Drop an audio file here or click to browse"
                        }
                        p {
                            style: "color: #888; margin-bottom: 1rem;",
                            "Supported formats: MP3, WAV, M4A, FLAC, OGG"
                        }
                        button {
                            style: "
                                background: #007bff;
                                color: white;
                                border: none;
                                padding: 0.75rem 1.5rem;
                                border-radius: 5px;
                                cursor: pointer;
                                font-size: 1rem;
                            ",
                            "Choose File"
                        }
                    }
                }
            }

            // Current file info
            if let Some(ref path) = state_read.file_path {
                div {
                    style: "
                        background: #e9ecef;
                        padding: 1rem;
                        border-radius: 5px;
                        margin-bottom: 2rem;
                    ",
                    h4 { "Selected File:" }
                    p { 
                        style: "margin: 0.5rem 0; word-break: break-all;",
                        "{path.display()}"
                    }
                }
            }

            // Error display
            if let Some(ref error) = state_read.error {
                div {
                    style: "
                        background: #f8d7da;
                        color: #721c24;
                        padding: 1rem;
                        border-radius: 5px;
                        margin-bottom: 2rem;
                        border: 1px solid #f5c6cb;
                    ",
                    h4 { "❌ Error" }
                    p { "{error}" }
                }
            }

            // Transcription result
            if let Some(ref result) = state_read.result {
                TranscriptionResult { result: result.clone(), state }
            }
        }
    }
}

#[component]
fn TranscriptionResult(result: String, state: Signal<TranscriptionState>) -> Element {
    let result_for_copy = result.clone();
    let copy_to_clipboard = move |_| {
        let result = result_for_copy.clone();
        spawn(async move {
            copy_text_to_clipboard(&result).await;
        });
    };

    let clear_results = move |_| {
        state.write().result = None;
        state.write().error = None;
        state.write().file_path = None;
    };

    rsx! {
        div {
            style: "
                background: #d4edda;
                color: #155724;
                padding: 1.5rem;
                border-radius: 5px;
                border: 1px solid #c3e6cb;
            ",
            h4 { 
                style: "margin-bottom: 1rem;",
                "✅ Transcription Result"
            }
            div {
                style: "
                    background: white;
                    padding: 1rem;
                    border-radius: 5px;
                    border: 1px solid #ddd;
                    white-space: pre-wrap;
                    font-family: Georgia, serif;
                    line-height: 1.6;
                    color: #333;
                ",
                "{result}"
            }
            div {
                style: "margin-top: 1rem;",
                button {
                    style: "
                        background: #28a745;
                        color: white;
                        border: none;
                        padding: 0.5rem 1rem;
                        border-radius: 3px;
                        cursor: pointer;
                        margin-right: 0.5rem;
                    ",
                    onclick: copy_to_clipboard,
                    "Copy to Clipboard"
                }
                button {
                    style: "
                        background: #6c757d;
                        color: white;
                        border: none;
                        padding: 0.5rem 1rem;
                        border-radius: 3px;
                        cursor: pointer;
                    ",
                    onclick: clear_results,
                    "Clear"
                }
            }
        }
    }
}

// Platform-specific file dialog implementation
async fn open_file_dialog() -> Option<PathBuf> {
    #[cfg(feature = "desktop")]
    {
        use rfd::AsyncFileDialog;
        
        if let Some(file) = AsyncFileDialog::new()
            .add_filter("Audio Files", &["mp3", "wav", "m4a", "flac", "ogg"])
            .pick_file()
            .await
        {
            return Some(file.path().to_path_buf());
        }
    }
    
    #[cfg(not(feature = "desktop"))]
    {
        // Web/mobile implementation - for now just log a warning
        tracing::warn!("File dialog not implemented for this platform yet");
    }
    
    None
}

// Platform-specific dropped file handling (placeholder for future implementation)
#[allow(dead_code)]
async fn handle_dropped_file(file_name: &str) -> Option<PathBuf> {
    #[cfg(feature = "desktop")]
    {
        // For desktop, the file name is typically the full path
        let path = PathBuf::from(file_name);
        if path.exists() {
            return Some(path);
        }
        None
    }
    
    #[cfg(not(feature = "desktop"))]
    {
        // Web/mobile would handle File objects differently
        tracing::warn!("Drag and drop not fully implemented for this platform yet");
        // For now, just try to create a path from the filename
        // In a real web implementation, this would handle File objects
        let path = PathBuf::from(file_name);
        Some(path)
    }
}

// Platform-specific clipboard implementation
async fn copy_text_to_clipboard(text: &str) {
    #[cfg(feature = "desktop")]
    {
        use copypasta::{ClipboardContext, ClipboardProvider};
        
        if let Ok(mut ctx) = ClipboardContext::new() {
            let _ = ctx.set_contents(text.to_string());
        }
    }
    
    #[cfg(not(feature = "desktop"))]
    {
        // Web/mobile clipboard implementation - just log for now
        tracing::warn!("Clipboard not implemented for this platform yet: {}", text.len());
    }
}

async fn transcribe_file(path: PathBuf, mut state: Signal<TranscriptionState>) {
    // Update state to show transcribing
    state.write().is_transcribing = true;
    state.write().error = None;
    state.write().result = None;

    let result = transcribe_file_impl(path).await;

    match result {
        Ok(text) => {
            state.write().result = Some(text);
            state.write().is_transcribing = false;
        }
        Err(e) => {
            state.write().error = Some(format!("Transcription failed: {}", e));
            state.write().is_transcribing = false;
        }
    }
}

// Server-side implementation using purr-core directly
#[cfg(feature = "server")]
async fn transcribe_file_impl(path: PathBuf) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // Create transcription config
    let config = TranscriptionConfig::new()
        .with_gpu(true)
        .with_sample_rate(16000);

    // Perform transcription
    let result = transcribe_audio_file(&path, Some(config)).await?;
    Ok(result.text)
}

// Client-side implementation using server function (when api crate is available)
#[cfg(feature = "api")]
#[cfg(not(feature = "server"))]
async fn transcribe_file_impl(path: PathBuf) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // For web clients, call the server function
    let file_path = path.to_string_lossy().to_string();
    
    match transcribe_audio(file_path, None, true).await {
        Ok(result) => Ok(result.text),
        Err(e) => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Server function failed: {}", e)
        )))
    }
}

// Fallback implementation when neither server nor api features are available
#[cfg(not(any(feature = "server", feature = "api")))]
async fn transcribe_file_impl(_path: PathBuf) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    Err(Box::new(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Transcription not available - no server or API access configured"
    )))
}