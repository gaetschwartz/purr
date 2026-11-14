//! transcribe.cpp as a Purr engine.

use std::path::Path;

use purr_common::model::engine::EngineId;
use purr_core::{
    accel::build_info::BuildInfo,
    dev::Device,
    engine::{Engine, OpenSpec, decoder::WindowDecoder, error::OpenError, speech::SpeechDetector},
};

use crate::{
    build_info::build_info,
    devices::list_devices,
    error::TranscribeEngineError,
    logging,
    runtime::{self, Runtime, Unavailable},
};

/// transcribe.cpp, with the ggml it was built with.
#[derive(Debug, Clone, Copy, Default)]
pub struct TranscribeEngine;

impl TranscribeEngine {
    /// Loads transcribe.cpp if that has not happened yet, and says whether it can be used.
    pub fn availability(&self) -> Result<&'static Runtime, &'static Unavailable> {
        runtime::get()
    }
}

impl Engine for TranscribeEngine {
    fn id(&self) -> EngineId {
        EngineId::TranscribeCpp
    }

    fn build_info(&self) -> BuildInfo {
        build_info()
    }

    fn devices(&self) -> Vec<Device> {
        match runtime::get() {
            Ok(_) => list_devices(),
            Err(_) => Vec::new(),
        }
    }

    fn open(&self, _spec: &OpenSpec<'_>) -> Result<Box<dyn WindowDecoder>, OpenError> {
        Err(TranscribeEngineError::CannotDecodeYet.into())
    }

    fn open_speech_detector(&self, _model: &Path) -> Result<Box<dyn SpeechDetector>, OpenError> {
        Err(TranscribeEngineError::CannotDecodeYet.into())
    }

    fn install_logging(&self) {
        logging::request();
        if let Some(Ok(_)) = runtime::loaded() {
            logging::install_if_requested();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_engine_is_transcribe_cpp() {
        assert_eq!(TranscribeEngine.id(), EngineId::TranscribeCpp);
    }

    #[test]
    fn opening_a_model_says_the_engine_cannot_decode_yet() {
        let error = TranscribeEngine
            .open_speech_detector(Path::new("model.gguf"))
            .err();
        assert!(matches!(error, Some(OpenError::Engine(_))));
    }
}
