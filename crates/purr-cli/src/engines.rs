//! The engines this binary is built with.

use std::sync::Arc;

use purr_core::engine::Engine;
use purr_engine_whisper::engine::WhisperEngine;

/// Every engine, for transcriptions to pick the one their model names.
pub fn all() -> Vec<Arc<dyn Engine>> {
    vec![Arc::new(WhisperEngine)]
}

/// The engine whose devices `purr sys` and `--device` describe.
pub const fn hardware() -> WhisperEngine {
    WhisperEngine
}
