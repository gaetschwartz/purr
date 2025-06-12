//! This crate contains all shared fullstack server functions.
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

/// Echo the user input on the server.
#[server(Echo)]
pub async fn echo(input: String) -> Result<String, ServerFnError> {
    Ok(input)
}

/// Transcription result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub processing_time: f64,
    pub audio_duration: f32,
    pub language: Option<String>,
}

/// Transcribe audio file on the server.
#[server(TranscribeAudio)]
pub async fn transcribe_audio(
    file_path: String,
    language: Option<String>,
    use_gpu: bool,
) -> Result<TranscriptionResult, ServerFnError> {
    use purr_core::{transcribe_audio_file, TranscriptionConfig};
    use std::path::PathBuf;

    tracing::info!("Transcribing audio file: {}", file_path);

    // Create transcription config
    let mut config = TranscriptionConfig::new()
        .with_gpu(use_gpu)
        .with_sample_rate(16000);

    if let Some(lang) = language {
        config = config.with_language(lang);
    }

    // Perform transcription
    let path = PathBuf::from(file_path);
    match transcribe_audio_file(&path, Some(config)).await {
        Ok(result) => Ok(TranscriptionResult {
            text: result.text,
            processing_time: result.processing_time,
            audio_duration: result.audio_duration,
            language: result.language,
        }),
        Err(e) => {
            tracing::error!("Transcription failed: {}", e);
            Err(ServerFnError::new(format!("Transcription failed: {}", e)))
        }
    }
}
