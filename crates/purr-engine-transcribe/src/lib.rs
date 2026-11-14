//! transcribe.cpp, through its Rust bindings, as a Purr speech recognition engine.
//!
//! [`engine::TranscribeEngine`] is the whole surface the pipeline needs. The engine cannot decode
//! yet: today it reports the backends and devices of its own ggml, kept separate from the ggml
//! that whisper.cpp links statically.

mod build_info;
mod devices;
pub mod engine;
mod error;
mod logging;
pub mod runtime;

pub const PKG_NAME: &str = env!("CARGO_PKG_NAME");
