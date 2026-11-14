//! transcribe.cpp as a Purr engine.

use std::path::Path;

use purr_common::{
    accel::{EngineUnavailable, LoadedEngine},
    model::engine::EngineId,
};
use purr_core::{
    accel::{backend::AccelBackend, build_info::BuildInfo},
    dev::Device,
    engine::{Engine, OpenSpec, decoder::WindowDecoder, error::OpenError, speech::SpeechDetector},
};
use transcribe_cpp::{Backend, Model, ModelOptions};

use crate::{
    build_info::build_info,
    decoder::ParakeetDecoder,
    devices::list_devices,
    error::{TranscribeEngineError, open_error},
    logging,
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
        Unavailable::Backends(_) => EngineUnavailable::NoBackend {
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
        runtime::get().map_err(TranscribeEngineError::Library)?;
        match spec.candidate.backend {
            AccelBackend::Cpu => {}
            AccelBackend::CoreMl
            | AccelBackend::Cuda
            | AccelBackend::Hip
            | AccelBackend::Metal
            | AccelBackend::Sycl
            | AccelBackend::Vulkan => return Err(OpenError::NoDevice),
        }
        let options = ModelOptions {
            backend: Backend::Cpu,
            ..ModelOptions::default()
        };
        let model = Model::load_with(spec.model, &options)
            .map_err(|error| open_error(error, &spec.candidate.name))?;
        Ok(Box::new(ParakeetDecoder::new(&model, spec.threads)?))
    }

    fn open_speech_detector(&self, model: &Path) -> Result<Box<dyn SpeechDetector>, OpenError> {
        runtime::get().map_err(TranscribeEngineError::Library)?;
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
