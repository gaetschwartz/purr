//! Developer mode on the desktop: models loaded from a file, and the readings of the panel.

use purr_common::{
    accel::LastRun,
    developer::{
        model::{DeveloperModel, DeveloperSource, NativeModel},
        settings::DeveloperSettings,
        snapshot::{NativeDevice, NativeSnapshot, Peaked},
    },
    language::support::LanguageSupport,
    model::{
        descriptor::{ModelDescriptor, Weights},
        engine::EngineId,
        profile::ModelProfile,
        quantization::Quantization,
    },
    platform::ModelInfo,
};
use purr_core::engine::Engine;
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

static PEAK_WORKING_SET: AtomicU64 = AtomicU64::new(0);
static PEAK_VIRTUAL: AtomicU64 = AtomicU64::new(0);

/// The model file a developer model names and the engine that loads it.
pub struct NativeChoice {
    pub path: PathBuf,
    pub engine: EngineId,
}

#[must_use]
pub fn choice(settings: &DeveloperSettings, id: &str) -> Option<(NativeChoice, u64)> {
    let model = settings.model(id)?;
    let DeveloperSource::Native(native) = &model.source else {
        return None;
    };
    let size = std::fs::metadata(&native.path).map_or(0, |meta| meta.len());
    Some((
        NativeChoice {
            path: native.path.clone(),
            engine: EngineId::loading(native.format),
        },
        size,
    ))
}

fn describe(model: &DeveloperModel, native: &NativeModel) -> ModelDescriptor {
    ModelDescriptor {
        engine: EngineId::loading(native.format),
        family: model.family,
        format: native.format,
        weights: Weights::Quantized(Quantization::F16),
        profile: ModelProfile::browser(LanguageSupport::Any),
    }
}

/// The developer models of the settings as the catalog lists them; a model whose file is gone
/// is listed as not installed.
#[must_use]
pub fn infos(settings: &DeveloperSettings) -> Vec<ModelInfo> {
    settings
        .active_models()
        .iter()
        .filter_map(|model| {
            let DeveloperSource::Native(native) = &model.source else {
                return None;
            };
            let metadata = std::fs::metadata(&native.path);
            let mut info = ModelInfo::new(
                model.id.as_str().to_owned(),
                model.name.clone(),
                format!("Loaded from {}", native.path.display()),
                metadata.as_ref().map_or(0, std::fs::Metadata::len),
                metadata.is_ok(),
                describe(model, native),
            )
            .from_developer();
            if metadata.is_ok() {
                info.mark_installed(native.path.clone());
            }
            Some(info)
        })
        .collect()
}

fn reported(bytes: usize) -> Option<u64> {
    (bytes > 0).then_some(bytes as u64)
}

fn peaked(current: u64, peak: &AtomicU64, reset: bool) -> Peaked {
    if reset {
        peak.store(current, Ordering::Relaxed);
    }
    let highest = peak.fetch_max(current, Ordering::Relaxed).max(current);
    Peaked {
        current,
        peak: highest,
    }
}

/// What the engines and the process report right now.
#[must_use]
pub fn snapshot(
    engines: &[Arc<dyn Engine>],
    last_run: Option<LastRun>,
    reset_peaks: bool,
) -> NativeSnapshot {
    let devices = engines
        .iter()
        .flat_map(|engine| {
            let id = engine.id();
            engine
                .devices()
                .into_iter()
                .map(move |device| NativeDevice {
                    engine: id,
                    name: device.name,
                    description: device.description,
                    device_id: device.device_id,
                    backend: device.backend.map(purr_common::accel::AccelBackend::from),
                    device_type: device.tpe,
                    memory_free: reported(device.vram_free),
                    memory_total: reported(device.vram_total),
                })
        })
        .collect();
    let process = memory_stats::memory_stats();
    NativeSnapshot {
        devices,
        last_run,
        process_memory: process
            .map(|stats| peaked(stats.physical_mem as u64, &PEAK_WORKING_SET, reset_peaks)),
        process_virtual: process
            .map(|stats| peaked(stats.virtual_mem as u64, &PEAK_VIRTUAL, reset_peaks)),
    }
}
