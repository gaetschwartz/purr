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

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::{
        developer::model::DeveloperModelId,
        model::{
            family::{ModelFamily, WhisperSize},
            format::ModelFormat,
        },
        platform::ModelOrigin,
    };

    fn settings_with(path: PathBuf, format: ModelFormat) -> (DeveloperSettings, String) {
        let model = DeveloperModel {
            id: DeveloperModelId::from_unique("0123456789ab"),
            name: "mine.bin".to_owned(),
            family: ModelFamily::Whisper(WhisperSize::Tiny),
            source: DeveloperSource::Native(NativeModel { path, format }),
        };
        let id = model.id.as_str().to_owned();
        (
            DeveloperSettings {
                enabled: true,
                session_options: None,
                models: vec![model],
            },
            id,
        )
    }

    #[test]
    fn a_ggml_file_runs_on_whisper_cpp_and_a_gguf_file_on_transcribe_cpp() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("model.bin");
        std::fs::write(&file, b"lmgg").unwrap();
        for (format, engine) in [
            (ModelFormat::GgmlBin, EngineId::WhisperCpp),
            (ModelFormat::Gguf, EngineId::TranscribeCpp),
        ] {
            let (settings, id) = settings_with(file.clone(), format);
            let (chosen, size) = choice(&settings, &id).unwrap();
            assert_eq!(chosen.engine, engine);
            assert_eq!(chosen.path, file);
            assert_eq!(size, 4);
        }
    }

    #[test]
    fn a_developer_model_is_listed_while_its_file_is_there_and_only_in_developer_mode() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("model.bin");
        std::fs::write(&file, b"lmgg").unwrap();
        let (mut settings, id) = settings_with(file.clone(), ModelFormat::GgmlBin);
        let listed = infos(&settings);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, id);
        assert_eq!(listed[0].origin, ModelOrigin::Developer);
        assert!(listed[0].is_installed);
        std::fs::remove_file(&file).unwrap();
        assert!(!infos(&settings)[0].is_installed);
        settings.enabled = false;
        assert_eq!(infos(&settings), Vec::<ModelInfo>::new());
        assert!(choice(&settings, &id).is_none());
    }

    #[test]
    fn a_peak_follows_the_highest_value_until_it_is_reset() {
        let peak = AtomicU64::new(0);
        assert_eq!(peaked(10, &peak, false).peak, 10);
        assert_eq!(
            peaked(4, &peak, false),
            Peaked {
                current: 4,
                peak: 10
            }
        );
        assert_eq!(
            peaked(4, &peak, true),
            Peaked {
                current: 4,
                peak: 4
            }
        );
    }

    #[test]
    fn the_snapshot_reports_this_process_even_without_engines() {
        let snapshot = snapshot(&[], None, false);
        assert_eq!(snapshot.devices, Vec::new());
        assert!(snapshot.last_run.is_none());
        assert!(
            snapshot
                .process_memory
                .is_some_and(|memory| memory.current > 0)
        );
    }
}
