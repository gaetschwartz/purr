//! The engines this binary is built with.

use std::sync::Arc;

use purr_core::engine::Engine;
use purr_engine_whisper::engine::WhisperEngine;

use crate::engine_report::EngineReport;

/// Every engine, for transcriptions to pick the one their model names.
pub fn all() -> Vec<Arc<dyn Engine>> {
    #[cfg(feature = "transcribe")]
    {
        vec![
            Arc::new(WhisperEngine),
            Arc::new(purr_engine_transcribe::engine::TranscribeEngine),
        ]
    }
    #[cfg(not(feature = "transcribe"))]
    {
        vec![Arc::new(WhisperEngine)]
    }
}

/// The engine whose devices `purr sys` and `--device` describe.
pub const fn hardware() -> WhisperEngine {
    WhisperEngine
}

/// What `purr sys` reports about the engines other than [`hardware`]. Loads their libraries.
pub fn reports() -> Vec<EngineReport> {
    #[cfg(feature = "transcribe")]
    {
        vec![crate::engine_report::transcribe()]
    }
    #[cfg(not(feature = "transcribe"))]
    {
        Vec::new()
    }
}
