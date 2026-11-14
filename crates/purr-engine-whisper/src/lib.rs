//! whisper.cpp, through whisper-rs, as a Purr speech recognition engine.
//!
//! [`engine::WhisperEngine`] is the whole public surface the pipeline needs; the rest is how it
//! talks to whisper.cpp and its ggml.

mod build_info;
mod coreml;
mod decoder;
mod devices;
pub mod engine;
mod error;
mod inventory;
mod language;
pub mod logging;
mod probe;
mod segments;
mod vad;
pub mod words;

pub const PKG_NAME: &str = env!("CARGO_PKG_NAME");
