//! The devices of this machine across every engine: one entry per physical device, listing the
//! engines that can run on it.
//!
//! Each engine has its own ggml and so its own device list; a device's index is a position in
//! one engine's list. What the engines share is a device's key (backend, name, occurrence), which
//! is what the user's saved order speaks of.

use purr_common::{devices::DeviceKey, model::engine::EngineId, platform::DeviceType};
use serde::Serialize;

use super::{
    backend::AccelBackend,
    devices::{DetectedDevice, priority},
    engine_report::EngineReport,
    report::UnavailableReason,
};

/// One engine's view of a device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Sighting {
    pub engine: EngineId,
    /// The key the engine's own device list gives it.
    pub key: DeviceKey,
    /// The device's position in the engine's list; `None` for a device no inventory lists.
    pub index: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HardwareDevice {
    /// Unique among the merged devices; the key of the first engine that listed the device.
    pub key: DeviceKey,
    pub backend: AccelBackend,
    pub name: String,
    pub pci_id: Option<String>,
    pub kind: Option<DeviceType>,
    /// Every engine that lists the device, in the order the engines were given.
    pub seen_by: Vec<Sighting>,
}

impl HardwareDevice {
    #[must_use]
    pub fn sighting(&self, engine: EngineId) -> Option<&Sighting> {
        self.seen_by
            .iter()
            .find(|sighting| sighting.engine == engine)
    }

    #[must_use]
    pub fn first_sighting(&self) -> Option<&Sighting> {
        self.seen_by.first()
    }

    /// Whether `device`, as another engine lists it, is this physical device.
    ///
    /// The backend and name must agree. When both report a PCI id it decides, so identical cards
    /// listed in a different order by two engines still pair up; otherwise the position among
    /// devices of that name does. The processor is one device whatever it is called.
    fn is_same_as(&self, device: &DetectedDevice) -> bool {
        if self.backend != device.backend {
            return false;
        }
        if self.backend == AccelBackend::Cpu {
            return true;
        }
        if self.name != device.name {
            return false;
        }
        match (&self.pci_id, &device.pci_id) {
            (Some(ours), Some(theirs)) => ours.eq_ignore_ascii_case(theirs),
            (None, _) | (_, None) => self.key.occurrence == device.key.occurrence,
        }
    }
}

fn unique_key(key: &DeviceKey, merged: &[HardwareDevice]) -> DeviceKey {
    let taken = |candidate: &DeviceKey| merged.iter().any(|device| device.key == *candidate);
    let mut candidate = key.clone();
    while taken(&candidate) {
        candidate.occurrence = candidate.occurrence.saturating_add(1);
        if candidate.occurrence == u8::MAX {
            break;
        }
    }
    candidate
}

/// The devices of every loaded engine in `reports`, each listed once, in the usual order: by
/// backend priority, and within a backend the order of the first engine that listed it.
#[must_use]
pub fn merge(reports: &[EngineReport]) -> Vec<HardwareDevice> {
    let mut merged: Vec<HardwareDevice> = Vec::new();
    for report in reports {
        for device in &report.detected {
            let sighting = Sighting {
                engine: report.engine,
                key: device.key.clone(),
                index: device.index,
            };
            let same = merged
                .iter_mut()
                .find(|known| known.is_same_as(device) && known.sighting(report.engine).is_none());
            if let Some(known) = same {
                known.seen_by.push(sighting);
                continue;
            }
            let at = merged
                .iter()
                .position(|known| priority(known.backend) > priority(device.backend))
                .unwrap_or(merged.len());
            let key = unique_key(&device.key, &merged);
            merged.insert(
                at,
                HardwareDevice {
                    key,
                    backend: device.backend,
                    name: device.name.clone(),
                    pci_id: device.pci_id.clone(),
                    kind: device.kind,
                    seen_by: vec![sighting],
                },
            );
        }
    }
    merged
}

fn rank(reason: &UnavailableReason) -> u8 {
    match reason {
        UnavailableReason::RuntimeLibraryMissing { .. } => 0,
        UnavailableReason::NoDevice => 1,
        UnavailableReason::NotBuiltOrNoDevice => 2,
        UnavailableReason::NotBuilt => 3,
    }
}

/// Whether `backend` can run on any loaded engine; when none can, the most telling reason.
pub fn backend_status(
    backend: AccelBackend,
    reports: &[EngineReport],
) -> Result<(), UnavailableReason> {
    let mut worst: Option<UnavailableReason> = None;
    for report in reports.iter().filter(|report| report.is_loaded()) {
        match report.backend_status(backend) {
            Ok(()) => return Ok(()),
            Err(reason) => {
                if worst
                    .as_ref()
                    .is_none_or(|known| rank(&reason) < rank(known))
                {
                    worst = Some(reason);
                }
            }
        }
    }
    Err(worst.unwrap_or(UnavailableReason::NotBuilt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accel::{
        devices::fixtures::{device, inventory, with_pci},
        engine_report::EngineReport,
    };
    use AccelBackend::{Cpu, Cuda, Vulkan};
    use purr_common::accel::{EngineAvailability, EngineUnavailable, LoadedEngine};

    const WHISPER: EngineId = EngineId::WhisperCpp;
    const TRANSCRIBE: EngineId = EngineId::TranscribeCpp;

    fn loaded(
        engine: EngineId,
        devices: Vec<crate::dev::Device>,
        built: &[AccelBackend],
    ) -> EngineReport {
        EngineReport::new(
            engine,
            EngineAvailability::Loaded(LoadedEngine {
                version: None,
                directory: None,
            }),
            inventory(devices, built),
        )
    }

    fn engines_of(merged: &HardwareDevice) -> Vec<EngineId> {
        merged.seen_by.iter().map(|s| s.engine).collect()
    }

    #[test]
    fn the_same_gpu_listed_by_two_engines_is_one_device_seen_by_both() {
        let whisper = loaded(
            WHISPER,
            vec![
                with_pci(device(0, "RTX", Vulkan, DeviceType::Gpu), "0000:01:00.0"),
                device(1, "Processor", Cpu, DeviceType::Cpu),
            ],
            &[Vulkan, Cpu],
        );
        let transcribe = loaded(
            TRANSCRIBE,
            vec![
                device(0, "Processor", Cpu, DeviceType::Cpu),
                with_pci(device(1, "RTX", Vulkan, DeviceType::Gpu), "0000:01:00.0"),
            ],
            &[Vulkan, Cpu],
        );
        let merged = merge(&[whisper, transcribe]);
        assert_eq!(
            merged
                .iter()
                .map(|d| (d.backend, engines_of(d)))
                .collect::<Vec<_>>(),
            [
                (Vulkan, vec![WHISPER, TRANSCRIBE]),
                (Cpu, vec![WHISPER, TRANSCRIBE])
            ]
        );
        assert_eq!(
            merged
                .first()
                .map(|d| d.seen_by.iter().map(|s| s.index).collect::<Vec<_>>()),
            Some(vec![Some(0), Some(1)])
        );
    }

    #[test]
    fn a_device_one_engine_lacks_is_seen_by_the_other_alone() {
        let whisper = loaded(
            WHISPER,
            vec![
                device(0, "RTX", Cuda, DeviceType::Gpu),
                device(1, "Processor", Cpu, DeviceType::Cpu),
            ],
            &[Cuda, Cpu],
        );
        let transcribe = loaded(
            TRANSCRIBE,
            vec![device(0, "Processor", Cpu, DeviceType::Cpu)],
            &[Cpu],
        );
        let merged = merge(&[whisper, transcribe]);
        assert_eq!(
            merged
                .iter()
                .map(|d| (d.backend, engines_of(d)))
                .collect::<Vec<_>>(),
            [(Cuda, vec![WHISPER]), (Cpu, vec![WHISPER, TRANSCRIBE])]
        );
    }

    #[test]
    fn the_processor_is_one_device_whatever_each_engine_calls_it() {
        let whisper = loaded(
            WHISPER,
            vec![device(0, "AMD Ryzen 9 7950X", Cpu, DeviceType::Cpu)],
            &[Cpu],
        );
        let transcribe = loaded(
            TRANSCRIBE,
            vec![device(0, "CPU", Cpu, DeviceType::Cpu)],
            &[Cpu],
        );
        let merged = merge(&[whisper, transcribe]);
        assert_eq!(merged.len(), 1);
        assert_eq!(
            merged.first().map(|d| d.key.clone()),
            Some(DeviceKey::cpu())
        );
        assert_eq!(
            merged.first().map(|d| d.name.as_str()),
            Some("AMD Ryzen 9 7950X")
        );
    }

    #[test]
    fn identical_cards_pair_by_pci_id_when_the_engines_order_them_differently() {
        let card = |id, pci| with_pci(device(id, "Same Card", Vulkan, DeviceType::Gpu), pci);
        let whisper = loaded(
            WHISPER,
            vec![card(0, "0000:01:00.0"), card(1, "0000:02:00.0")],
            &[Vulkan],
        );
        let transcribe = loaded(
            TRANSCRIBE,
            vec![card(0, "0000:02:00.0"), card(1, "0000:01:00.0")],
            &[Vulkan],
        );
        let merged = merge(&[whisper, transcribe]);
        assert_eq!(
            merged
                .iter()
                .map(|d| {
                    (
                        d.key.occurrence,
                        d.seen_by.iter().map(|s| s.index).collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>(),
            [(0, vec![Some(0), Some(1)]), (1, vec![Some(1), Some(0)])]
        );
    }

    #[test]
    fn merged_keys_stay_unique_when_a_pci_id_splits_equal_keys() {
        let whisper = loaded(
            WHISPER,
            vec![with_pci(
                device(0, "Same Card", Vulkan, DeviceType::Gpu),
                "0000:01:00.0",
            )],
            &[Vulkan],
        );
        let transcribe = loaded(
            TRANSCRIBE,
            vec![with_pci(
                device(0, "Same Card", Vulkan, DeviceType::Gpu),
                "0000:02:00.0",
            )],
            &[Vulkan],
        );
        let merged = merge(&[whisper, transcribe]);
        assert_eq!(
            merged.iter().map(|d| d.key.occurrence).collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(
            merged.iter().map(engines_of).collect::<Vec<_>>(),
            [vec![WHISPER], vec![TRANSCRIBE]]
        );
    }

    #[test]
    fn a_device_without_a_pci_id_pairs_by_name_and_position() {
        let igpu = |id| device(id, "Radeon", Vulkan, DeviceType::Igpu);
        let whisper = loaded(WHISPER, vec![igpu(0), igpu(1)], &[Vulkan]);
        let transcribe = loaded(TRANSCRIBE, vec![igpu(0)], &[Vulkan]);
        let merged = merge(&[whisper, transcribe]);
        assert_eq!(
            merged.iter().map(engines_of).collect::<Vec<_>>(),
            [vec![WHISPER, TRANSCRIBE], vec![WHISPER]]
        );
    }

    #[test]
    fn a_device_known_to_one_pci_id_only_pairs_by_position() {
        let whisper = loaded(
            WHISPER,
            vec![with_pci(
                device(0, "Radeon", Vulkan, DeviceType::Igpu),
                "0000:00:02.0",
            )],
            &[Vulkan],
        );
        let transcribe = loaded(
            TRANSCRIBE,
            vec![device(0, "Radeon", Vulkan, DeviceType::Igpu)],
            &[Vulkan],
        );
        assert_eq!(merge(&[whisper, transcribe]).len(), 1);
    }

    #[test]
    fn an_engine_that_is_not_loaded_contributes_no_device() {
        let whisper = loaded(
            WHISPER,
            vec![device(0, "Processor", Cpu, DeviceType::Cpu)],
            &[Cpu],
        );
        let transcribe = EngineReport::new(
            TRANSCRIBE,
            EngineAvailability::Unavailable(EngineUnavailable::NoBackend {
                detail: String::new(),
            }),
            inventory(vec![device(0, "Processor", Cpu, DeviceType::Cpu)], &[Cpu]),
        );
        let merged = merge(&[whisper, transcribe]);
        assert_eq!(
            merged.iter().map(engines_of).collect::<Vec<_>>(),
            [vec![WHISPER]]
        );
    }

    #[test]
    fn devices_of_a_later_engine_go_after_the_ones_of_better_backends_only() {
        let whisper = loaded(
            WHISPER,
            vec![
                device(0, "RTX", Cuda, DeviceType::Gpu),
                device(1, "Processor", Cpu, DeviceType::Cpu),
            ],
            &[Cuda, Cpu],
        );
        let transcribe = loaded(
            TRANSCRIBE,
            vec![
                device(0, "Radeon", Vulkan, DeviceType::Igpu),
                device(1, "Processor", Cpu, DeviceType::Cpu),
            ],
            &[Vulkan, Cpu],
        );
        assert_eq!(
            merge(&[whisper, transcribe])
                .iter()
                .map(|d| d.backend)
                .collect::<Vec<_>>(),
            [Cuda, Vulkan, Cpu]
        );
    }

    #[test]
    fn a_backend_runs_when_any_loaded_engine_runs_it() {
        let whisper = loaded(WHISPER, vec![], &[Vulkan, Cpu]);
        let transcribe = loaded(
            TRANSCRIBE,
            vec![device(0, "Radeon", Vulkan, DeviceType::Igpu)],
            &[Vulkan],
        );
        assert_eq!(backend_status(Vulkan, &[whisper, transcribe]), Ok(()));
    }

    #[test]
    fn the_most_telling_reason_wins_when_no_engine_runs_a_backend() {
        let built = loaded(WHISPER, vec![], &[Cuda]);
        let not_built = loaded(TRANSCRIBE, vec![], &[Cpu]);
        assert_eq!(
            backend_status(Cuda, &[not_built, built]),
            Err(UnavailableReason::NoDevice)
        );
    }
}
