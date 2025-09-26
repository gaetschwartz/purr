//! WebAssembly platform implementation for Purr transcription engine
//!
//! This crate provides a WebAssembly-compatible implementation of the Platform trait
//! for running Purr in web browsers. It handles file processing, model management,
//! and transcription coordination using web APIs.

use wasm_bindgen::prelude::*;

// Re-export main types
pub use error::{AudioFormatError, StorageError, WebError, WebGpuError, WebResult, WorkerError};
pub use model::{DownloadProgress, ModelInfo, ModelStorageStats, WebModelManager};
pub use platform::PlatformImpl;
pub use storage::{FileMetadata, WebStorage};
pub use transcription::{
    get_supported_formats, start_transcription_process, validate_audio_file, AudioMetadata,
    AudioProcessingConfig, AudioProcessingProgress, AudioTranscriptionProcessor,
};
pub use worker::{TranscriptionConfig, TranscriptionWorker};

// Module declarations
mod error;
mod model;
pub mod platform;
mod storage;
mod transcription;
mod worker;

// Test utilities (only available in test builds)
#[cfg(test)]
pub mod webgpu_test_utils;

// Initialize the web platform
#[wasm_bindgen(start)]
pub fn init() -> Result<(), JsValue> {
    // print pretty errors in wasm https://github.com/rustwasm/console_error_panic_hook
    // This is not needed for tracing_wasm to work, but it is a common tool for getting proper error line numbers for panics.
    console_error_panic_hook::set_once();

    // Add this line:
    tracing_wasm::set_as_global_default();

    console_log!("Purr WASM Platform initialized");
    Ok(())
}

// Import console.log for debugging
pub(crate) mod console {
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console)]
        pub(crate) fn log(a: js_sys::Array);
        #[wasm_bindgen(js_namespace = console, js_name = error)]
        pub(crate) fn error(a: js_sys::Array);
        #[wasm_bindgen(js_namespace = console, js_name = warn)]
        pub(crate) fn warn(a: js_sys::Array);
        #[wasm_bindgen(js_namespace = console, js_name = info)]
        pub(crate) fn info(a: js_sys::Array);
        #[wasm_bindgen(js_namespace = console, js_name = debug)]
        pub(crate) fn debug(a: js_sys::Array);
        #[wasm_bindgen(js_namespace = console, js_name = inspect)]
        pub(crate) fn inspect(s: &JsValue);
    }
}

// Macro for console logging (public for modules)
#[macro_export]
macro_rules! console_log {
    ($($e:expr),*) => {
        let array = ::js_sys::Array::new();
        $(array.push(&wasm_bindgen::JsValue::from($e));)*
        $crate::console::log(array);
    }
}

#[macro_export]
macro_rules! console_error {
    ($($e:expr),*) => {
        let array = ::js_sys::Array::new();
        $(array.push(&wasm_bindgen::JsValue::from($e));)*
        $crate::console::error(array);
    }
}

#[macro_export]
macro_rules! console_warn {
    ($($e:expr),*) => {
        let array = ::js_sys::Array::new();
        $(array.push(&wasm_bindgen::JsValue::from($e));)*
        $crate::console::warn(array);
    }
}
#[macro_export]
macro_rules! console_info {
    ($($e:expr),*) => {
        let array = ::js_sys::Array::new();
        $(array.push(&wasm_bindgen::JsValue::from($e));)*
        $crate::console::info(array);
    }
}
#[macro_export]
macro_rules! console_debug {
    ($($e:expr),*) => {
        let array = ::js_sys::Array::new();
        $(array.push(&wasm_bindgen::JsValue::from($e));)*
        $crate::console::debug(array);
    }
}

#[macro_export]
macro_rules! console_inspect {
    ($s:expr) => {
        $crate::console::inspect(&wasm_bindgen::JsValue::from($s));
    };
}
