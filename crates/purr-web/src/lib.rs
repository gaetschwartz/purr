//! Web platform implementation for Purr transcription engine
//!
//! This crate provides a WebAssembly-compatible implementation of the Platform trait
//! for running Purr in web browsers. It handles file processing, model management,
//! and transcription coordination using web APIs.

use wasm_bindgen::prelude::*;

// Import console.log for debugging
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

// Macro for console logging (public for modules)
#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

// Re-export main types
pub use model::{WebModelManager, ModelInfo, DownloadProgress, ModelStorageStats};
pub use platform::PlatformImpl;
pub use storage::{WebStorage, FileMetadata};
pub use worker::{TranscriptionWorker, TranscriptionConfig};
pub use transcription::{
    start_transcription_process, validate_audio_file, get_supported_formats,
    AudioTranscriptionProcessor, AudioProcessingConfig, AudioMetadata, AudioProcessingProgress,
};
pub use error::{WebError, WebResult, WorkerError, StorageError, AudioFormatError, WebGpuError};

// Module declarations
mod error;
mod model;
pub mod platform;
mod storage;
mod worker;
mod transcription;

// Test utilities (only available in test builds)
#[cfg(test)]
pub mod webgpu_test_utils;

// NOTE: Removed panic hooks - let Dioxus handle panic management
// Dioxus provides its own panic handling and WASM initialization

// Initialize the web platform
#[wasm_bindgen(start)]
pub fn init() {
    // Let Dioxus handle panic hooks and WASM setup
    console_log!("Purr Web Platform initialized - Dioxus manages panic handling");
}
