//! transcribe.cpp as a Purr engine.

use std::path::Path;

use purr_common::{
    accel::{EngineUnavailable, LoadedEngine},
    model::engine::EngineId,
};
use purr_core::{
    accel::build_info::BuildInfo,
    dev::Device,
    engine::{Engine, OpenSpec, decoder::WindowDecoder, error::OpenError, speech::SpeechDetector},
};
use transcribe_cpp::Model;

use crate::{
    build_info::build_info,
    decoder::TranscribeDecoder,
    devices::list_devices,
    error::{TranscribeEngineError, open_error},
    logging,
    placement::model_options,
    runtime::{self, Unavailable},
    vad::SileroDetector,
};

/// transcribe.cpp, with the ggml it was built with.
#[derive(Debug, Clone, Copy, Default)]
pub struct TranscribeEngine;

fn unavailable(why: &Unavailable) -> EngineUnavailable {
    match why {
        Unavailable::LibraryMissing { library, searched } => EngineUnavailable::LibraryMissing {
            library: library.clone(),
            searched: searched.clone(),
        },
        Unavailable::LibraryUnloadable { library, directory } => {
            EngineUnavailable::LibraryUnloadable {
                library: library.clone(),
                directory: directory.clone(),
            }
        }
        Unavailable::Backends { .. } => EngineUnavailable::NoBackend {
            detail: why.to_string(),
        },
    }
}

impl Engine for TranscribeEngine {
    fn id(&self) -> EngineId {
        EngineId::TranscribeCpp
    }

    fn availability(&self) -> Result<LoadedEngine, EngineUnavailable> {
        runtime::get()
            .map(|runtime| LoadedEngine {
                version: Some(runtime.version.clone()),
                directory: runtime.directory.clone(),
            })
            .map_err(|why| unavailable(&why))
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

    fn open(&self, spec: &OpenSpec<'_>) -> Result<Box<dyn WindowDecoder>, OpenError> {
        runtime::get().map_err(|source| TranscribeEngineError::Library { source })?;
        let options = model_options(spec.candidate, spec.devices)?;
        let model = Model::load_with(spec.model, &options)
            .map_err(|error| open_error(error.into(), spec.candidate))?;
        let decoder = TranscribeDecoder::new(&model, spec.threads)
            .map_err(|error| open_error(error, spec.candidate))?;
        Ok(Box::new(decoder))
    }

    fn open_speech_detector(&self, model: &Path) -> Result<Box<dyn SpeechDetector>, OpenError> {
        runtime::get().map_err(|source| TranscribeEngineError::Library { source })?;
        Ok(Box::new(SileroDetector::load(model)?))
    }

    fn install_logging(&self) {
        logging::request();
        if runtime::loaded().is_some() {
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
}
