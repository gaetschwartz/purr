//! whisper.cpp, through whisper-rs, as the engine behind the pipeline.

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
