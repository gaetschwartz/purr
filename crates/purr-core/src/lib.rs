//! Purr's transcription core: audio input, models, hardware, and the pipeline that turns audio
//! into text with whichever engine runs the model.

pub mod accel;
pub mod audio;
pub mod config;
pub mod cpu;
pub mod dev;
pub mod engine;
pub mod error;
pub mod http;
pub mod input;
pub mod math;
pub mod model;
pub mod pipeline;
mod resample;
pub mod url;

pub use audio::{AudioChunk, AudioProcessor, AudioStream, SourceLength};
pub use dev::{Device, SystemInfo};
pub use error::{Error, Result};
pub use model::ModelManager;
pub use url::{HttpStreamer, UrlContentInfo, UrlStreamConfig};

pub const PKG_NAME: &str = env!("CARGO_PKG_NAME");
