//! The engines this binary is built with.

use std::sync::Arc;

use purr_common::model::engine::EngineId;
use purr_core::engine::Engine;
use purr_engine_whisper::engine::WhisperEngine;

/// The engines a full build of Purr has, whether or not this binary was built with them.
pub const KNOWN: [EngineId; 2] = [EngineId::WhisperCpp, EngineId::TranscribeCpp];

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

/// The ids of the engines this binary is built with.
pub fn built() -> Vec<EngineId> {
    all().iter().map(|engine| engine.id()).collect()
}
