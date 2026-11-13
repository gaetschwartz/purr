//! The user's device list as screens show it: the saved order merged with what this computer
//! has, and the edits the Hardware page and the Transcribe screen make to it.

use crate::utils::byte_size::ByteSize;
use purr_common::{
    accel::{
        AccelBackend, BackendReport, BackendStatus, BackendUnavailable, CpuFeature, CpuReport,
        SystemReport,
    },
    devices::{DeviceKey, DevicePreference},
    platform::{DeviceInfo, DeviceType},
    settings::PerformanceSettings,
};

/// What the system says about a device of the list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceState {
    Available,
    /// Detected, but its backend cannot run.
    Unavailable(BackendUnavailable),
    /// Saved, but not detected on this computer; the reason is its backend's, when it has one.
    NotFound(Option<BackendUnavailable>),
}

/// One line of the device list.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceRow {
    pub key: DeviceKey,
    pub backend: AccelBackend,
    /// The name the system reports, as it reports it.
    pub name: String,
    pub kind: Option<DeviceType>,
    pub detail: Option<String>,
    pub enabled: bool,
    pub state: DeviceState,
}

impl DeviceRow {
    /// Whether a transcription can be asked to run on it.
    #[must_use]
    pub const fn usable(&self) -> bool {
        matches!(self.state, DeviceState::Available)
    }
}

/// Where a device moves in the list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Up,
    Down,
}

pub fn device_name(device: &DeviceInfo) -> String {
    match device.description.as_deref() {
        Some(description) if !description.trim().is_empty() => description.to_owned(),
        Some(_) | None => device.name.clone(),
    }
}

fn cpu_detail(cpu: &CpuReport) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(threads) = cpu.logical_threads {
        parts.push(format!("{threads} threads"));
    }
    if !cpu.built_with.is_empty() {
        parts.push(CpuFeature::join_labels(&cpu.built_with));
    }
    let missing = cpu.missing_on_cpu();
    if !missing.is_empty() {
        parts.push(format!(
            "not on this CPU: {}",
            CpuFeature::join_labels(&missing)
        ));
    }
    (!parts.is_empty()).then(|| parts.join(" · "))
}

fn gpu_detail(device: &DeviceInfo) -> Option<String> {
    let memory = device
        .memory_total
        .filter(|total| *total > 0)
        .map(|total| format!("{} memory", ByteSize(total as u64)));
    let kind = match device.device_type {
        DeviceType::Gpu | DeviceType::Igpu => Some(device.device_type.label().to_owned()),
        DeviceType::Cpu | DeviceType::Accel | DeviceType::Unknown => None,
    };
    let parts: Vec<String> = [kind, memory].into_iter().flatten().collect();
    (!parts.is_empty()).then(|| parts.join(" · "))
}

struct Detected<'a> {
    report: &'a BackendReport,
    info: &'a DeviceInfo,
    key: &'a DeviceKey,
}

fn detected(system: &SystemReport) -> Vec<Detected<'_>> {
    system
        .backends
        .iter()
        .flat_map(|report| {
            report
                .devices
                .iter()
                .filter_map(move |info| info.key.as_ref().map(|key| Detected { report, info, key }))
        })
        .collect()
}

/// The keys of the devices `system` reports, in the order they are tried when nobody arranged
/// them.
#[must_use]
pub fn detected_keys(system: &SystemReport) -> Vec<DeviceKey> {
    detected(system)
        .into_iter()
        .map(|device| device.key.clone())
        .collect()
}

/// The device list for this computer: the saved order, the devices not yet saved after the saved
/// ones, and saved devices that are gone marked as such.
#[must_use]
pub fn device_rows(performance: &PerformanceSettings, system: &SystemReport) -> Vec<DeviceRow> {
    let found = detected(system);
    let keys: Vec<DeviceKey> = found.iter().map(|device| device.key.clone()).collect();
    performance
        .device_order()
        .effective(&keys)
        .into_iter()
        .map(|preference| {
            let key = preference.key;
            match found.iter().find(|device| *device.key == key) {
                Some(device) => DeviceRow {
                    backend: key.backend,
                    name: device_name(device.info),
                    kind: Some(device.info.device_type),
                    detail: match key.backend {
                        AccelBackend::Cpu => cpu_detail(&system.cpu),
                        AccelBackend::Cuda
                        | AccelBackend::Hip
                        | AccelBackend::CoreMl
                        | AccelBackend::Metal
                        | AccelBackend::Sycl
                        | AccelBackend::Vulkan => gpu_detail(device.info),
                    },
                    enabled: preference.enabled,
                    state: match &device.report.status {
                        BackendStatus::Available => DeviceState::Available,
                        BackendStatus::Unavailable(why) => DeviceState::Unavailable(why.clone()),
                    },
                    key,
                },
                None => DeviceRow {
                    backend: key.backend,
                    name: key.name.clone(),
                    kind: None,
                    detail: None,
                    enabled: preference.enabled,
                    state: DeviceState::NotFound(
                        system
                            .backends
                            .iter()
                            .find(|report| report.backend == key.backend)
                            .and_then(|report| match &report.status {
                                BackendStatus::Available => None,
                                BackendStatus::Unavailable(why) => Some(why.clone()),
                            }),
                    ),
                    key,
                },
            }
        })
        .collect()
}

/// How many rows from the top can be reached: the devices after the CPU never are, since the CPU
/// always loads.
#[must_use]
pub fn reachable_rows(rows: &[DeviceRow]) -> usize {
    rows.iter()
        .position(|row| row.key.is_cpu())
        .map_or(rows.len(), |cpu| cpu + 1)
}

/// The list as it is saved.
#[must_use]
pub fn saved_list(rows: &[DeviceRow]) -> Vec<DevicePreference> {
    rows.iter()
        .map(|row| DevicePreference {
            key: row.key.clone(),
            enabled: row.enabled,
        })
        .collect()
}

/// `list` with `key` one place earlier or later; the ends stay put.
#[must_use]
pub fn moved(list: &[DevicePreference], key: &DeviceKey, direction: Move) -> Vec<DevicePreference> {
    let mut next = list.to_vec();
    let Some(from) = next.iter().position(|device| device.key == *key) else {
        return next;
    };
    let to = match direction {
        Move::Up => from.checked_sub(1),
        Move::Down => Some(from + 1).filter(|to| *to < next.len()),
    };
    if let Some(to) = to {
        next.swap(from, to);
    }
    next
}

/// `list` with `key` switched on or off; the processor stays on.
#[must_use]
pub fn toggled(list: &[DevicePreference], key: &DeviceKey, enabled: bool) -> Vec<DevicePreference> {
    list.iter()
        .map(|device| DevicePreference {
            key: device.key.clone(),
            enabled: if device.key == *key && !key.is_cpu() {
                enabled
            } else {
                device.enabled
            },
        })
        .collect()
}

/// What a run can be asked to start on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceChoice {
    /// The order in the settings.
    Automatic,
    Device(DeviceKey),
}

/// The choices the Transcribe screen offers: the settings order, then every device that is
/// switched on and can run, in list order.
#[must_use]
pub fn choices(rows: &[DeviceRow]) -> Vec<DeviceChoice> {
    std::iter::once(DeviceChoice::Automatic)
        .chain(
            rows.iter()
                .filter(|row| row.enabled && row.usable())
                .map(|row| DeviceChoice::Device(row.key.clone())),
        )
        .collect()
}

/// The device a run starts on when `picked` was chosen: `None` when it is no longer a choice.
#[must_use]
pub fn override_for(picked: Option<&DeviceKey>, rows: &[DeviceRow]) -> Option<DeviceKey> {
    let picked = picked?;
    choices(rows).into_iter().find_map(|choice| match choice {
        DeviceChoice::Device(key) if key == *picked => Some(key),
        DeviceChoice::Device(_) | DeviceChoice::Automatic => None,
    })
}

/// The row for `key`.
#[must_use]
pub fn row_of<'a>(rows: &'a [DeviceRow], key: &DeviceKey) -> Option<&'a DeviceRow> {
    rows.iter().find(|row| row.key == *key)
}

/// "Vulkan · AMD Radeon(TM) Graphics": the backend and the device as the system names it.
#[must_use]
pub fn row_phrase(row: &DeviceRow) -> String {
    format!("{} · {}", row.backend.label(), row.name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::devices::DeviceDefaults;

    const RTX: &str = "NVIDIA GeForce RTX 2070 SUPER";
    const RADEON: &str = "AMD Radeon(TM) Graphics";
    const RYZEN: &str = "AMD Ryzen 9 7950X 16-Core Processor";

    fn info(
        name: &str,
        device_type: DeviceType,
        backend: AccelBackend,
        memory: usize,
    ) -> DeviceInfo {
        DeviceInfo {
            id: 0,
            name: "ggml0".to_owned(),
            description: Some(name.to_owned()),
            device_type,
            memory_free: None,
            memory_total: Some(memory),
            capabilities: None,
            key: Some(DeviceKey::new(backend, name, 0)),
        }
    }

    fn report(
        backend: AccelBackend,
        status: BackendStatus,
        devices: Vec<DeviceInfo>,
    ) -> BackendReport {
        BackendReport {
            backend,
            status,
            devices,
        }
    }

    fn system() -> SystemReport {
        SystemReport {
            backends: vec![
                report(
                    AccelBackend::Cuda,
                    BackendStatus::Available,
                    vec![info(RTX, DeviceType::Gpu, AccelBackend::Cuda, 8 << 30)],
                ),
                report(
                    AccelBackend::Metal,
                    BackendStatus::Unavailable(BackendUnavailable::NotBuilt),
                    vec![],
                ),
                report(
                    AccelBackend::Vulkan,
                    BackendStatus::Available,
                    vec![
                        info(RTX, DeviceType::Gpu, AccelBackend::Vulkan, 8 << 30),
                        info(RADEON, DeviceType::Igpu, AccelBackend::Vulkan, 0),
                    ],
                ),
                report(
                    AccelBackend::Cpu,
                    BackendStatus::Available,
                    vec![info(RYZEN, DeviceType::Cpu, AccelBackend::Cpu, 0)],
                ),
            ],
            cpu: CpuReport {
                name: Some(RYZEN.to_owned()),
                logical_threads: Some(32),
                built_with: vec![CpuFeature::Avx2],
                detected: None,
            },
            last_run: None,
        }
    }

    fn cuda() -> DeviceKey {
        DeviceKey::new(AccelBackend::Cuda, RTX, 0)
    }

    fn vulkan() -> DeviceKey {
        DeviceKey::new(AccelBackend::Vulkan, RTX, 0)
    }

    fn radeon() -> DeviceKey {
        DeviceKey::new(AccelBackend::Vulkan, RADEON, 0)
    }

    fn keys(rows: &[DeviceRow]) -> Vec<DeviceKey> {
        rows.iter().map(|row| row.key.clone()).collect()
    }

    fn performance(list: Vec<DevicePreference>) -> PerformanceSettings {
        let mut performance = PerformanceSettings::default();
        performance.set_devices(list);
        performance
    }

    fn pref(key: DeviceKey, enabled: bool) -> DevicePreference {
        DevicePreference { key, enabled }
    }

    fn cpu(built_with: &[CpuFeature], detected: Option<&[CpuFeature]>) -> CpuReport {
        CpuReport {
            name: None,
            logical_threads: Some(8),
            built_with: built_with.to_vec(),
            detected: detected.map(<[CpuFeature]>::to_vec),
        }
    }

    #[test]
    fn the_cpu_detail_lists_what_the_build_uses() {
        let both = [CpuFeature::Avx2, CpuFeature::Fma];
        assert_eq!(
            cpu_detail(&cpu(&both, Some(&both))),
            Some("8 threads · AVX2, FMA".to_owned())
        );
    }

    #[test]
    fn the_cpu_detail_flags_a_mismatch_between_the_build_and_the_processor() {
        assert_eq!(
            cpu_detail(&cpu(
                &[CpuFeature::Avx2, CpuFeature::Avx512F],
                Some(&[CpuFeature::Avx2, CpuFeature::Bmi2])
            )),
            Some("8 threads · AVX2, AVX512F · not on this CPU: AVX512F".to_owned())
        );
    }

    #[test]
    fn the_cpu_detail_makes_no_claim_where_the_processor_cannot_be_probed() {
        assert_eq!(
            cpu_detail(&cpu(&[CpuFeature::Avx2], None)),
            Some("8 threads · AVX2".to_owned())
        );
    }

    #[test]
    fn a_fresh_profile_lists_every_device_in_the_usual_order_all_on() {
        let rows = device_rows(&PerformanceSettings::default(), &system());
        assert_eq!(
            rows.iter()
                .map(|row| (row.key.clone(), row.enabled))
                .collect::<Vec<_>>(),
            [
                (cuda(), true),
                (vulkan(), true),
                (radeon(), true),
                (DeviceKey::cpu(), true)
            ]
        );
    }

    #[test]
    fn a_row_carries_the_reported_name_the_kind_and_the_detail() {
        let rows = device_rows(&PerformanceSettings::default(), &system());
        let summary: Vec<_> = rows
            .iter()
            .map(|row| (row.name.as_str(), row.kind, row.detail.clone()))
            .collect();
        assert_eq!(
            summary,
            [
                (
                    RTX,
                    Some(DeviceType::Gpu),
                    Some("discrete GPU · 8.0 GiB memory".to_owned())
                ),
                (
                    RTX,
                    Some(DeviceType::Gpu),
                    Some("discrete GPU · 8.0 GiB memory".to_owned())
                ),
                (
                    RADEON,
                    Some(DeviceType::Igpu),
                    Some("integrated GPU".to_owned())
                ),
                (
                    RYZEN,
                    Some(DeviceType::Cpu),
                    Some("32 threads · AVX2".to_owned())
                ),
            ]
        );
    }

    #[test]
    fn the_saved_order_and_switches_win() {
        let saved = performance(vec![
            pref(radeon(), true),
            pref(cuda(), false),
            pref(DeviceKey::cpu(), true),
            pref(vulkan(), true),
        ]);
        let rows = device_rows(&saved, &system());
        assert_eq!(
            rows.iter()
                .map(|row| (row.key.clone(), row.enabled))
                .collect::<Vec<_>>(),
            [
                (radeon(), true),
                (cuda(), false),
                (DeviceKey::cpu(), true),
                (vulkan(), true)
            ]
        );
    }

    #[test]
    fn a_saved_device_that_is_gone_stays_as_not_found_with_its_name() {
        let gone = DeviceKey::new(AccelBackend::Vulkan, "Unplugged eGPU", 0);
        let metal = DeviceKey::new(AccelBackend::Metal, "Apple M3", 0);
        let saved = performance(vec![
            pref(gone.clone(), true),
            pref(metal.clone(), true),
            pref(cuda(), true),
            pref(DeviceKey::cpu(), true),
        ]);
        let rows = device_rows(&saved, &system());
        assert_eq!(
            rows.iter()
                .take(2)
                .map(|row| (row.name.as_str(), row.state.clone()))
                .collect::<Vec<_>>(),
            [
                ("Unplugged eGPU", DeviceState::NotFound(None)),
                (
                    "Apple M3",
                    DeviceState::NotFound(Some(BackendUnavailable::NotBuilt))
                ),
            ]
        );
        assert_eq!(rows.len(), 6);
    }

    #[test]
    fn a_device_detected_since_the_list_was_saved_is_added_on_after_the_gpus() {
        let saved = performance(vec![pref(cuda(), true), pref(DeviceKey::cpu(), true)]);
        assert_eq!(
            keys(&device_rows(&saved, &system())),
            [cuda(), vulkan(), radeon(), DeviceKey::cpu()]
        );
    }

    #[test]
    fn migrated_defaults_without_gpus_show_the_gpus_switched_off() {
        let mut migrated = PerformanceSettings::default();
        migrated.device_defaults = DeviceDefaults {
            gpus_enabled: false,
            backends_first: Vec::new(),
        };
        let rows = device_rows(&migrated, &system());
        assert_eq!(
            rows.iter().map(|row| row.enabled).collect::<Vec<_>>(),
            [false, false, false, true]
        );
    }

    #[test]
    fn migrated_defaults_with_a_forced_backend_list_its_devices_first() {
        let mut migrated = PerformanceSettings::default();
        migrated.device_defaults = DeviceDefaults {
            gpus_enabled: true,
            backends_first: vec![AccelBackend::Vulkan],
        };
        assert_eq!(
            keys(&device_rows(&migrated, &system())),
            [vulkan(), radeon(), cuda(), DeviceKey::cpu()]
        );
    }

    #[test]
    fn a_detected_device_of_a_backend_that_cannot_run_is_unavailable() {
        let mut broken = system();
        broken.backends[0].status = BackendStatus::Unavailable(BackendUnavailable::NoDevice);
        let rows = device_rows(&PerformanceSettings::default(), &broken);
        assert_eq!(
            rows.first().map(|row| row.state.clone()),
            Some(DeviceState::Unavailable(BackendUnavailable::NoDevice))
        );
    }

    #[test]
    fn the_devices_after_the_cpu_are_out_of_reach() {
        let rows = device_rows(&PerformanceSettings::default(), &system());
        assert_eq!(reachable_rows(&rows), 4);
        let cpu_first = performance(vec![pref(DeviceKey::cpu(), true), pref(cuda(), true)]);
        assert_eq!(reachable_rows(&device_rows(&cpu_first, &system())), 1);
        assert_eq!(reachable_rows(&[]), 0);
    }

    #[test]
    fn a_device_moves_one_place_and_the_ends_stay_put() {
        let list = vec![
            pref(cuda(), true),
            pref(vulkan(), true),
            pref(DeviceKey::cpu(), true),
        ];
        assert_eq!(
            keys_of(&moved(&list, &vulkan(), Move::Up)),
            [vulkan(), cuda(), DeviceKey::cpu()]
        );
        assert_eq!(
            keys_of(&moved(&list, &vulkan(), Move::Down)),
            [cuda(), DeviceKey::cpu(), vulkan()]
        );
        assert_eq!(moved(&list, &cuda(), Move::Up), list);
        assert_eq!(moved(&list, &DeviceKey::cpu(), Move::Down), list);
        assert_eq!(moved(&list, &radeon(), Move::Up), list);
    }

    fn keys_of(list: &[DevicePreference]) -> Vec<DeviceKey> {
        list.iter().map(|device| device.key.clone()).collect()
    }

    #[test]
    fn the_cpu_can_move_up_so_it_runs_before_the_gpus() {
        let list = vec![pref(cuda(), true), pref(DeviceKey::cpu(), true)];
        assert_eq!(
            keys_of(&moved(&list, &DeviceKey::cpu(), Move::Up)),
            [DeviceKey::cpu(), cuda()]
        );
    }

    #[test]
    fn a_switch_changes_one_device_and_never_the_cpu() {
        let list = vec![pref(cuda(), true), pref(DeviceKey::cpu(), true)];
        assert_eq!(
            toggled(&list, &cuda(), false),
            [pref(cuda(), false), pref(DeviceKey::cpu(), true)]
        );
        assert_eq!(toggled(&list, &DeviceKey::cpu(), false), list);
    }

    #[test]
    fn saving_the_rows_keeps_gone_devices() {
        let gone = DeviceKey::new(AccelBackend::Metal, "Apple M3", 0);
        let saved = performance(vec![
            pref(gone.clone(), false),
            pref(DeviceKey::cpu(), true),
        ]);
        let rows = device_rows(&saved, &system());
        assert!(keys_of(&saved_list(&rows)).contains(&gone));
        assert_eq!(saved_list(&rows).first(), Some(&pref(gone, false)));
    }

    #[test]
    fn a_run_may_pick_automatic_or_any_enabled_device_that_can_run() {
        let saved = performance(vec![
            pref(cuda(), false),
            pref(vulkan(), true),
            pref(DeviceKey::new(AccelBackend::Metal, "Apple M3", 0), true),
            pref(DeviceKey::cpu(), true),
        ]);
        let rows = device_rows(&saved, &system());
        assert_eq!(
            choices(&rows),
            [
                DeviceChoice::Automatic,
                DeviceChoice::Device(vulkan()),
                DeviceChoice::Device(radeon()),
                DeviceChoice::Device(DeviceKey::cpu()),
            ]
        );
    }

    #[test]
    fn a_picked_device_that_was_switched_off_or_removed_is_forgotten() {
        let saved = performance(vec![pref(cuda(), false), pref(DeviceKey::cpu(), true)]);
        let rows = device_rows(&saved, &system());
        assert_eq!(override_for(Some(&cuda()), &rows), None);
        assert_eq!(
            override_for(Some(&DeviceKey::new(AccelBackend::Cuda, "Other", 0)), &rows),
            None
        );
        assert_eq!(override_for(Some(&vulkan()), &rows), Some(vulkan()));
        assert_eq!(override_for(None, &rows), None);
    }

    #[test]
    fn a_row_is_phrased_as_backend_then_device() {
        let rows = device_rows(&PerformanceSettings::default(), &system());
        assert_eq!(
            rows.iter().map(row_phrase).collect::<Vec<_>>(),
            [
                "CUDA · NVIDIA GeForce RTX 2070 SUPER",
                "Vulkan · NVIDIA GeForce RTX 2070 SUPER",
                "Vulkan · AMD Radeon(TM) Graphics",
                "CPU · AMD Ryzen 9 7950X 16-Core Processor",
            ]
        );
    }
}
