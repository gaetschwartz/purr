//! What one engine offers on this machine: whether it runs, what it was built with and the
//! devices it lists. Each engine has its own ggml, so each has its own inventory.

use std::sync::Arc;

use purr_common::{
    accel::{EngineAvailability, EngineSummary},
    model::engine::EngineId,
};

use crate::{dev::Device, engine::Engine};

use super::{
    backend::AccelBackend,
    devices::{DetectedDevice, detect},
    inventory::Inventory,
    plan::backend_status,
    report::UnavailableReason,
    selector::EngineDevices,
    vulkan_probe,
};

#[derive(Debug, Clone)]
pub struct EngineReport {
    pub engine: EngineId,
    pub availability: EngineAvailability,
    /// The engine's own devices and build; empty while the engine is not loaded.
    pub inventory: Inventory,
    /// The devices a transcription can run on, in the order they are tried when nobody arranged
    /// them; their indexes are positions in `inventory.devices`.
    pub detected: Vec<DetectedDevice>,
}

impl EngineReport {
    #[must_use]
    pub fn new(engine: EngineId, availability: EngineAvailability, inventory: Inventory) -> Self {
        let detected = match availability {
            EngineAvailability::Loaded(_) => detect(&inventory),
            EngineAvailability::NotBuilt | EngineAvailability::Unavailable(_) => Vec::new(),
        };
        Self {
            engine,
            availability,
            inventory,
            detected,
        }
    }

    #[must_use]
    pub fn not_built(engine: EngineId) -> Self {
        Self::new(engine, EngineAvailability::NotBuilt, Inventory::default())
    }

    /// Asks `engine` about itself. Loads its software and, once that works, lists its devices.
    #[must_use]
    pub fn of(engine: &dyn Engine) -> Self {
        match engine.availability() {
            Ok(loaded) => {
                let mut inventory = Inventory::detect(engine);
                vulkan_probe::attach_drivers(&mut inventory.devices);
                Self::new(engine.id(), EngineAvailability::Loaded(loaded), inventory)
            }
            Err(unavailable) => Self::new(
                engine.id(),
                EngineAvailability::Unavailable(unavailable),
                Inventory {
                    build: engine.build_info(),
                    ..Inventory::default()
                },
            ),
        }
    }

    #[must_use]
    pub const fn is_loaded(&self) -> bool {
        matches!(self.availability, EngineAvailability::Loaded(_))
    }

    #[must_use]
    pub fn summary(&self) -> EngineSummary {
        EngineSummary {
            engine: self.engine,
            availability: self.availability.clone(),
        }
    }

    /// The devices a selector is matched against for this engine.
    #[must_use]
    pub fn listed(&self) -> EngineDevices<'_> {
        EngineDevices {
            engine: self.engine,
            devices: &self.detected,
        }
    }

    /// The engine's device at `index` of its own list.
    #[must_use]
    pub fn device(&self, index: i32) -> Option<&Device> {
        self.inventory
            .devices
            .iter()
            .find(|device| device.id == index)
    }

    /// Whether `backend` can run on this engine, or why it cannot.
    pub fn backend_status(&self, backend: AccelBackend) -> Result<(), UnavailableReason> {
        backend_status(backend, &self.inventory, &self.detected)
    }
}

/// A report for each of `engines`, in order, then one marked not built for each of `expected`
/// that `engines` lacks.
#[must_use]
pub fn reports(engines: &[Arc<dyn Engine>], expected: &[EngineId]) -> Vec<EngineReport> {
    let present = engines
        .iter()
        .map(|engine| EngineReport::of(engine.as_ref()));
    let missing = expected
        .iter()
        .filter(|id| engines.iter().all(|engine| engine.id() != **id))
        .map(|id| EngineReport::not_built(*id));
    present.chain(missing).collect()
}

#[cfg(test)]
pub(crate) mod fakes {
    use std::path::Path;

    use super::*;
    use crate::{
        accel::{build_info::BuildInfo, devices::fixtures::inventory},
        engine::{OpenSpec, decoder::WindowDecoder, error::OpenError, speech::SpeechDetector},
    };
    use purr_common::accel::{EngineUnavailable, LoadedEngine};

    /// An engine that lists the devices it is given, or is unavailable.
    pub struct Scripted {
        pub id: EngineId,
        pub outcome: Result<LoadedEngine, EngineUnavailable>,
        pub devices: Vec<Device>,
        pub built: Vec<AccelBackend>,
    }

    impl Engine for Scripted {
        fn id(&self) -> EngineId {
            self.id
        }

        fn availability(&self) -> Result<LoadedEngine, EngineUnavailable> {
            self.outcome.clone()
        }

        fn build_info(&self) -> BuildInfo {
            inventory(Vec::new(), &self.built).build
        }

        fn devices(&self) -> Vec<Device> {
            self.devices.clone()
        }

        fn open(&self, _spec: &OpenSpec<'_>) -> Result<Box<dyn WindowDecoder>, OpenError> {
            Err(OpenError::NoDevice)
        }

        fn open_speech_detector(
            &self,
            _model: &Path,
        ) -> Result<Box<dyn SpeechDetector>, OpenError> {
            Err(OpenError::NoDevice)
        }

        fn install_logging(&self) {}
    }

    pub fn loaded(version: &str) -> Result<LoadedEngine, EngineUnavailable> {
        Ok(LoadedEngine {
            version: Some(version.to_owned()),
            directory: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::fakes::{Scripted, loaded};
    use super::*;
    use crate::accel::devices::fixtures::device;
    use AccelBackend::{Cpu, Vulkan};
    use purr_common::{accel::EngineUnavailable, platform::DeviceType};

    fn scripted(
        id: EngineId,
        outcome: Result<purr_common::accel::LoadedEngine, EngineUnavailable>,
        devices: Vec<Device>,
    ) -> Arc<dyn Engine> {
        Arc::new(Scripted {
            id,
            outcome,
            devices,
            built: vec![Cpu, Vulkan],
        })
    }

    fn processor() -> Device {
        device(0, "Processor", Cpu, DeviceType::Cpu)
    }

    #[test]
    fn a_loaded_engine_lists_its_own_devices_by_its_own_index() {
        let engines = [scripted(
            EngineId::WhisperCpp,
            loaded("1.0"),
            vec![
                device(0, "Radeon", Vulkan, DeviceType::Igpu),
                device(1, "Processor", Cpu, DeviceType::Cpu),
            ],
        )];
        let [report] = &reports(&engines, &[])[..] else {
            panic!("expected one report");
        };
        assert!(report.is_loaded());
        assert_eq!(
            report
                .detected
                .iter()
                .map(|d| (d.backend, d.index))
                .collect::<Vec<_>>(),
            [(Vulkan, Some(0)), (Cpu, Some(1))]
        );
    }

    #[test]
    fn an_engine_that_cannot_load_lists_no_device_and_keeps_its_reason() {
        let why = EngineUnavailable::LibraryMissing {
            library: "transcribe.dll".to_owned(),
            searched: Vec::new(),
        };
        let engines = [scripted(
            EngineId::TranscribeCpp,
            Err(why.clone()),
            vec![processor()],
        )];
        let [report] = &reports(&engines, &[])[..] else {
            panic!("expected one report");
        };
        assert_eq!(report.availability, EngineAvailability::Unavailable(why));
        assert_eq!(report.detected, []);
        assert_eq!(report.inventory.devices.len(), 0);
    }

    #[test]
    fn an_expected_engine_that_is_not_in_the_list_is_not_built() {
        let engines = [scripted(
            EngineId::WhisperCpp,
            loaded("1.0"),
            vec![processor()],
        )];
        let found = reports(&engines, &[EngineId::WhisperCpp, EngineId::TranscribeCpp]);
        assert_eq!(
            found
                .iter()
                .map(|report| (report.engine, report.availability.clone()))
                .collect::<Vec<_>>(),
            [
                (
                    EngineId::WhisperCpp,
                    EngineAvailability::Loaded(purr_common::accel::LoadedEngine {
                        version: Some("1.0".to_owned()),
                        directory: None
                    })
                ),
                (EngineId::TranscribeCpp, EngineAvailability::NotBuilt),
            ]
        );
    }

    #[test]
    fn a_device_is_found_by_its_index_in_the_engines_own_list() {
        let engines = [scripted(
            EngineId::WhisperCpp,
            loaded("1.0"),
            vec![processor()],
        )];
        let [report] = &reports(&engines, &[])[..] else {
            panic!("expected one report");
        };
        assert_eq!(
            report.device(0).map(|device| device.description.as_str()),
            Some("Processor")
        );
        assert_eq!(report.device(5).map(|device| device.id), None);
    }
}
