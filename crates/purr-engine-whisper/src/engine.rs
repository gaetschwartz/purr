//! whisper.cpp as a Purr engine.

use std::path::Path;

use purr_common::model::engine::EngineId;
use tracing::debug;
use whisper_rs::{WhisperContext, WhisperContextParameters};

use purr_core::{
    accel::{backend::AccelBackend, build_info::BuildInfo, report::AttemptFailure},
    dev::Device,
    engine::{Engine, OpenSpec, decoder::WindowDecoder, error::OpenError, speech::SpeechDetector},
};

use crate::{
    build_info::build_info, coreml, decoder::WhisperDecoder, devices::locate,
    inventory::list_devices, logging::install_logging_hooks, probe::probe_gpu, vad::SileroDetector,
};

/// whisper.cpp, with the ggml it was built with.
#[derive(Debug, Clone, Copy, Default)]
pub struct WhisperEngine;

impl Engine for WhisperEngine {
    fn id(&self) -> EngineId {
        EngineId::WhisperCpp
    }

    fn build_info(&self) -> BuildInfo {
        build_info()
    }

    fn devices(&self) -> Vec<Device> {
        list_devices()
    }

    fn open(&self, spec: &OpenSpec<'_>) -> Result<Box<dyn WindowDecoder>, OpenError> {
        let mut params = WhisperContextParameters::default();
        match spec.candidate.backend {
            AccelBackend::Cpu => {
                params.use_gpu(false);
            }
            AccelBackend::CoreMl => {
                let path = coreml::encoder_path(spec.model);
                if !path.exists() {
                    return Err(AttemptFailure::CoreMlModelMissing { path }.into());
                }
                params.use_gpu(false);
            }
            AccelBackend::Cuda
            | AccelBackend::Hip
            | AccelBackend::Metal
            | AccelBackend::Sycl
            | AccelBackend::Vulkan => {
                let gpu = locate(spec.candidate, spec.devices).ok_or(OpenError::NoDevice)?;
                probe_gpu(&gpu)?;
                params.use_gpu(true).gpu_device(gpu.ordinal);
            }
        }

        let context = WhisperContext::new_with_params(spec.model, params).map_err(|error| {
            debug!("whisper context creation failed: {error}");
            AttemptFailure::ContextInit
        })?;
        let eot = context.token_eot();
        let state = context.create_state().map_err(|error| {
            debug!("whisper state creation failed: {error}");
            AttemptFailure::StateInit
        })?;
        Ok(Box::new(WhisperDecoder::new(
            context,
            state,
            eot,
            spec.threads,
        )))
    }

    fn open_speech_detector(&self, model: &Path) -> Result<Box<dyn SpeechDetector>, OpenError> {
        Ok(Box::new(SileroDetector::load(model)?))
    }

    fn install_logging(&self) {
        install_logging_hooks();
    }
}
