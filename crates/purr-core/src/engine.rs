//! What a speech recognition engine provides to the pipeline.
//!
//! An engine loads models on the devices it enumerates and decodes windows of audio; the
//! pipeline decides which windows, in what order, and what of their text is final.

pub mod cancel;
pub mod decoder;
pub mod error;
pub mod hypothesis;
pub mod open;
pub mod speech;
pub mod text;

use std::path::Path;

use purr_common::{
    accel::{EngineUnavailable, LoadedEngine},
    model::engine::EngineId,
};

use crate::{
    accel::{build_info::BuildInfo, plan::Candidate},
    dev::Device,
};

use self::{decoder::WindowDecoder, error::OpenError, speech::SpeechDetector};

/// What an engine needs to load a model on one device.
#[derive(Debug, Clone, Copy)]
pub struct OpenSpec<'a> {
    pub model: &'a Path,
    pub candidate: &'a Candidate,
    /// The devices the engine listed for the plan the candidate comes from
    pub devices: &'a [Device],
    pub threads: usize,
}

/// One speech recognition engine built into this binary.
///
/// The runtime libraries of the GPU backends must be opened before an engine touches its
/// devices, which [`crate::accel::inventory::Inventory::detect`] does before it calls
/// [`Engine::devices`].
pub trait Engine: Send + Sync {
    fn id(&self) -> EngineId;

    /// Loads the engine's software if that has not happened yet, and says whether it can be used.
    ///
    /// An engine that is linked into the executable is always loaded.
    fn availability(&self) -> Result<LoadedEngine, EngineUnavailable> {
        Ok(LoadedEngine {
            version: None,
            directory: None,
        })
    }

    /// The backends and instruction sets the engine was compiled with.
    fn build_info(&self) -> BuildInfo;

    /// The devices the engine can run on, in its own order; a device's `id` is its index there.
    fn devices(&self) -> Vec<Device>;

    /// Loads `spec.model` on the device of `spec.candidate`.
    fn open(&self, spec: &OpenSpec<'_>) -> Result<Box<dyn WindowDecoder>, OpenError>;

    /// Loads the voice activity model at `model`.
    fn open_speech_detector(&self, model: &Path) -> Result<Box<dyn SpeechDetector>, OpenError>;

    /// Routes the engine's own log output into `tracing`.
    fn install_logging(&self);
}
