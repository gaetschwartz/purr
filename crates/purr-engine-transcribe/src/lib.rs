//! transcribe.cpp, through its Rust bindings, as a Purr speech recognition engine.
//!
//! [`engine::TranscribeEngine`] is the whole surface the pipeline needs. It runs the models
//! transcribe.cpp supports, such as NVIDIA Parakeet and Qwen3-ASR, on the processor or on a GPU,
//! on transcribe.cpp's own ggml, which is kept separate from the ggml that whisper.cpp links
//! statically.

mod build_info;
mod decoder;
mod devices;
pub mod engine;
mod error;
mod languages;
mod logging;
mod placement;
pub mod runtime;
mod transcript;
mod vad;

pub const PKG_NAME: &str = env!("CARGO_PKG_NAME");
