//! Client-side API for communicating with the backend server
//! This module provides HTTP and WebSocket communication for WASM builds

use serde::{Deserialize, Serialize};

/// Upload status for file uploads
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum UploadStatus {
    InProgress { bytes_received: usize },
    Completed { file_id: String },
    Error { message: String },
}

/// Transcription status updates
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
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
    Error {
        message: String,
    },
}
