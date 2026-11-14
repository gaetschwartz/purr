//! The engines this binary is built with.

use std::sync::Arc;

use purr_core::{engine::Engine, whisper::engine::WhisperEngine};

/// Every engine, for transcriptions to pick the one their model names.
pub fn all() -> Vec<Arc<dyn Engine>> {
    vec![Arc::new(WhisperEngine)]
}

/// The engine whose devices the compute settings list.
pub const fn hardware() -> WhisperEngine {
    WhisperEngine
}
