//! The ordered list of devices transcription tries, and how a saved list meets the devices
//! detected on this machine.

use crate::accel::AccelBackend;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Names one device across runs: the backend that drives it, the name the system reports and,
/// among devices of that backend with the same name, which one it is.
///
/// The processor is a singleton: every `Cpu` key is equal whatever the processor is called.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "StoredDeviceKey")]
pub struct DeviceKey {
    pub backend: AccelBackend,
    pub name: String,
    pub occurrence: u8,
}

#[derive(Deserialize)]
struct StoredDeviceKey {
    backend: AccelBackend,
    name: String,
    #[serde(default)]
    occurrence: u8,
}

impl From<StoredDeviceKey> for DeviceKey {
    fn from(stored: StoredDeviceKey) -> Self {
        Self::new(stored.backend, stored.name, stored.occurrence)
    }
}

impl DeviceKey {
    #[must_use]
    pub fn new(backend: AccelBackend, name: impl Into<String>, occurrence: u8) -> Self {
        match backend {
            AccelBackend::Cpu => Self::cpu(),
            AccelBackend::Cuda
            | AccelBackend::Hip
            | AccelBackend::CoreMl
            | AccelBackend::Metal
            | AccelBackend::Sycl
            | AccelBackend::Vulkan => Self {
                backend,
                name: name.into(),
                occurrence,
            },
        }
    }

    #[must_use]
    pub const fn cpu() -> Self {
        Self {
            backend: AccelBackend::Cpu,
            name: String::new(),
            occurrence: 0,
        }
    }

    #[must_use]
    pub const fn is_cpu(&self) -> bool {
        matches!(self.backend, AccelBackend::Cpu)
    }
}

/// Keys for devices given in a fixed order as (backend, reported name); devices of one backend
/// that share a name are told apart by their position among them.
#[must_use]
pub fn keys_in_order(devices: impl IntoIterator<Item = (AccelBackend, String)>) -> Vec<DeviceKey> {
    let mut seen: Vec<(AccelBackend, String, u8)> = Vec::new();
    devices
        .into_iter()
        .map(|(backend, name)| {
            let occurrence = match seen
                .iter_mut()
                .find(|(b, n, _)| *b == backend && *n == name)
            {
                Some((_, _, count)) => {
                    *count = count.saturating_add(1);
                    *count
                }
                None => {
                    seen.push((backend, name.clone(), 0));
                    0
                }
            };
            DeviceKey::new(backend, name, occurrence)
        })
        .collect()
}

/// A device in the user's list and whether it may be used.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DevicePreference {
    pub key: DeviceKey,
    pub enabled: bool,
}

impl DevicePreference {
    #[must_use]
    pub const fn enabled(key: DeviceKey) -> Self {
        Self { key, enabled: true }
    }
}

/// How devices nobody has arranged yet are ordered: set by a migrated setting or a command line
/// option, and forgotten once a list is saved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceDefaults {
    /// `false` leaves every device but the processor switched off.
    #[serde(default = "enabled")]
    pub gpus_enabled: bool,
    /// Backends whose devices come first, in this order; the rest follow in the usual order.
    #[serde(default)]
    pub backends_first: Vec<AccelBackend>,
}

const fn enabled() -> bool {
    true
}

impl Default for DeviceDefaults {
    fn default() -> Self {
        Self {
            gpus_enabled: true,
            backends_first: Vec::new(),
        }
    }
}

impl DeviceDefaults {
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// What transcription is asked to try, before the machine's devices are known.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DeviceOrder {
    devices: Vec<DevicePreference>,
    defaults: DeviceDefaults,
    first: Option<DeviceKey>,
}

impl DeviceOrder {
    /// Every detected device in the usual order.
    #[must_use]
    pub fn automatic() -> Self {
        Self::default()
    }

    /// The processor alone.
    #[must_use]
    pub fn cpu_only() -> Self {
        Self::saved(vec![DevicePreference::enabled(DeviceKey::cpu())])
    }

    /// The user's saved list.
    #[must_use]
    pub fn saved(devices: Vec<DevicePreference>) -> Self {
        Self {
            devices,
            defaults: DeviceDefaults::default(),
            first: None,
        }
    }

    /// Devices ordered by `defaults` until a list is saved.
    #[must_use]
    pub fn with_defaults(mut self, defaults: DeviceDefaults) -> Self {
        self.defaults = defaults;
        self
    }

    /// Tries `key` before the rest of the list, switched on for this order only.
    #[must_use]
    pub fn with_first(mut self, key: Option<DeviceKey>) -> Self {
        self.first = key;
        self
    }

    #[must_use]
    pub fn saved_devices(&self) -> &[DevicePreference] {
        &self.devices
    }

    #[must_use]
    pub const fn defaults(&self) -> &DeviceDefaults {
        &self.defaults
    }

    #[must_use]
    pub const fn first(&self) -> Option<&DeviceKey> {
        self.first.as_ref()
    }

    /// Whether only the processor will be used, as far as can be told without the detected
    /// devices. A browser, which detects none, asks this to choose between its GPU and its CPU.
    #[must_use]
    pub fn is_cpu_only(&self) -> bool {
        if let Some(first) = &self.first {
            return first.is_cpu();
        }
        if self.devices.is_empty() {
            return !self.defaults.gpus_enabled
                || self.defaults.backends_first.first() == Some(&AccelBackend::Cpu);
        }
        self.devices
            .iter()
            .find(|device| device.enabled || device.key.is_cpu())
            .is_none_or(|device| device.key.is_cpu())
    }

    /// The list to try on a machine with `detected` devices, given in the usual order.
    ///
    /// Saved devices that are not detected stay, devices not yet saved are added, the processor
    /// is always on, and the per-run first device leads.
    #[must_use]
    pub fn effective(&self, detected: &[DeviceKey]) -> Vec<DevicePreference> {
        let mut list = if self.devices.is_empty() {
            default_list(detected, &self.defaults)
        } else {
            merge_detected(&self.devices, detected)
        };
        for device in &mut list {
            if device.key.is_cpu() {
                device.enabled = true;
            }
        }
        if let Some(first) = &self.first {
            list.retain(|device| device.key != *first);
            list.insert(0, DevicePreference::enabled(first.clone()));
        }
        list
    }
}

fn default_list(detected: &[DeviceKey], defaults: &DeviceDefaults) -> Vec<DevicePreference> {
    let rank = |key: &DeviceKey| {
        defaults
            .backends_first
            .iter()
            .position(|backend| *backend == key.backend)
            .unwrap_or(defaults.backends_first.len())
    };
    let mut keys: Vec<&DeviceKey> = detected.iter().collect();
    keys.sort_by_key(|key| rank(key));
    keys.into_iter()
        .map(|key| DevicePreference {
            key: key.clone(),
            enabled: defaults.gpus_enabled || key.is_cpu(),
        })
        .collect()
}

/// `saved` with the detected devices it lacks added: after its last device that is not the
/// processor, or at the end when it has none, and the processor at the end.
#[must_use]
pub fn merge_detected(saved: &[DevicePreference], detected: &[DeviceKey]) -> Vec<DevicePreference> {
    let mut listed = HashSet::new();
    let mut list: Vec<DevicePreference> = saved
        .iter()
        .filter(|device| listed.insert(device.key.clone()))
        .cloned()
        .collect();
    for key in detected {
        if !listed.insert(key.clone()) {
            continue;
        }
        let position = if key.is_cpu() {
            list.len()
        } else {
            list.iter()
                .rposition(|device| !device.key.is_cpu())
                .map_or(list.len(), |last| last + 1)
        };
        list.insert(position, DevicePreference::enabled(key.clone()));
    }
    list
}

#[cfg(test)]
mod tests {
    use super::*;
    use AccelBackend::{Cpu, Cuda, Metal, Vulkan};

    fn key(backend: AccelBackend, name: &str) -> DeviceKey {
        DeviceKey::new(backend, name, 0)
    }

    fn on(key: DeviceKey) -> DevicePreference {
        DevicePreference::enabled(key)
    }

    fn off(key: DeviceKey) -> DevicePreference {
        DevicePreference {
            key,
            enabled: false,
        }
    }

    fn rtx() -> DeviceKey {
        key(Cuda, "RTX")
    }

    fn vulkan_rtx() -> DeviceKey {
        key(Vulkan, "RTX")
    }

    fn radeon() -> DeviceKey {
        key(Vulkan, "Radeon")
    }

    fn machine() -> Vec<DeviceKey> {
        vec![rtx(), vulkan_rtx(), radeon(), DeviceKey::cpu()]
    }

    #[test]
    fn identical_names_under_one_backend_are_told_apart_by_position() {
        let keys = keys_in_order([
            (Vulkan, "Radeon".to_owned()),
            (Cuda, "Radeon".to_owned()),
            (Vulkan, "Radeon".to_owned()),
            (Vulkan, "Other".to_owned()),
            (Vulkan, "Radeon".to_owned()),
        ]);
        assert_eq!(
            keys,
            [
                DeviceKey::new(Vulkan, "Radeon", 0),
                DeviceKey::new(Cuda, "Radeon", 0),
                DeviceKey::new(Vulkan, "Radeon", 1),
                DeviceKey::new(Vulkan, "Other", 0),
                DeviceKey::new(Vulkan, "Radeon", 2),
            ]
        );
    }

    #[test]
    fn the_same_gpu_under_two_backends_is_two_devices() {
        assert_ne!(rtx(), vulkan_rtx());
    }

    #[test]
    fn the_processor_is_one_device_whatever_it_is_called() {
        assert_eq!(
            DeviceKey::new(Cpu, "Ryzen 9", 0),
            DeviceKey::new(Cpu, "", 3)
        );
        assert_eq!(DeviceKey::new(Cpu, "Ryzen 9", 0), DeviceKey::cpu());
    }

    #[test]
    fn a_stored_processor_key_matches_whatever_name_it_was_saved_with() {
        let stored: DeviceKey =
            serde_json::from_str(r#"{"backend":"Cpu","name":"Old name","occurrence":2}"#).unwrap();
        assert_eq!(stored, DeviceKey::cpu());
    }

    #[test]
    fn a_stored_key_without_an_occurrence_is_the_first() {
        let stored: DeviceKey =
            serde_json::from_str(r#"{"backend":"Vulkan","name":"Radeon"}"#).unwrap();
        assert_eq!(stored, radeon());
    }

    #[test]
    fn nothing_saved_means_every_detected_device_in_the_given_order() {
        assert_eq!(
            DeviceOrder::automatic().effective(&machine()),
            [
                on(rtx()),
                on(vulkan_rtx()),
                on(radeon()),
                on(DeviceKey::cpu())
            ]
        );
    }

    #[test]
    fn migrated_defaults_without_gpus_leave_only_the_processor_on() {
        let order = DeviceOrder::automatic().with_defaults(DeviceDefaults {
            gpus_enabled: false,
            backends_first: Vec::new(),
        });
        assert_eq!(
            order.effective(&machine()),
            [
                off(rtx()),
                off(vulkan_rtx()),
                off(radeon()),
                on(DeviceKey::cpu())
            ]
        );
    }

    #[test]
    fn migrated_defaults_put_the_named_backends_first() {
        let order = DeviceOrder::automatic().with_defaults(DeviceDefaults {
            gpus_enabled: true,
            backends_first: vec![Vulkan],
        });
        assert_eq!(
            order.effective(&machine()),
            [
                on(vulkan_rtx()),
                on(radeon()),
                on(rtx()),
                on(DeviceKey::cpu())
            ]
        );
    }

    #[test]
    fn defaults_naming_the_processor_put_it_first() {
        let order = DeviceOrder::automatic().with_defaults(DeviceDefaults {
            gpus_enabled: true,
            backends_first: vec![Cpu],
        });
        assert_eq!(
            order.effective(&machine()).first(),
            Some(&on(DeviceKey::cpu()))
        );
    }

    #[test]
    fn defaults_are_ignored_once_a_list_is_saved() {
        let order = DeviceOrder::saved(vec![on(radeon()), on(DeviceKey::cpu())]).with_defaults(
            DeviceDefaults {
                gpus_enabled: false,
                backends_first: Vec::new(),
            },
        );
        assert_eq!(
            order.effective(&[radeon(), DeviceKey::cpu()]),
            [on(radeon()), on(DeviceKey::cpu())]
        );
    }

    #[test]
    fn a_saved_order_and_switches_are_kept() {
        let saved = vec![
            on(radeon()),
            off(rtx()),
            on(DeviceKey::cpu()),
            on(vulkan_rtx()),
        ];
        assert_eq!(
            DeviceOrder::saved(saved.clone()).effective(&machine()),
            saved
        );
    }

    #[test]
    fn a_saved_device_that_is_gone_stays_in_the_list() {
        let saved = vec![on(key(Metal, "M3")), on(radeon()), on(DeviceKey::cpu())];
        assert_eq!(
            DeviceOrder::saved(saved.clone()).effective(&[radeon(), DeviceKey::cpu()]),
            saved
        );
    }

    #[test]
    fn a_new_device_goes_after_the_last_saved_gpu() {
        let a = radeon();
        let b = rtx();
        assert_eq!(
            merge_detected(&[on(DeviceKey::cpu()), on(a.clone())], &[b.clone()]),
            [on(DeviceKey::cpu()), on(a.clone()), on(b.clone())]
        );
        assert_eq!(
            merge_detected(&[on(a.clone()), on(DeviceKey::cpu())], &[b.clone()]),
            [on(a), on(b.clone()), on(DeviceKey::cpu())]
        );
        assert_eq!(
            merge_detected(&[on(DeviceKey::cpu())], &[b.clone()]),
            [on(DeviceKey::cpu()), on(b)]
        );
    }

    #[test]
    fn several_new_devices_keep_the_detected_order() {
        assert_eq!(
            merge_detected(
                &[on(radeon()), on(DeviceKey::cpu())],
                &[rtx(), vulkan_rtx()]
            ),
            [
                on(radeon()),
                on(rtx()),
                on(vulkan_rtx()),
                on(DeviceKey::cpu())
            ]
        );
    }

    #[test]
    fn a_new_processor_goes_last() {
        assert_eq!(
            merge_detected(&[on(radeon())], &[DeviceKey::cpu()]),
            [on(radeon()), on(DeviceKey::cpu())]
        );
    }

    #[test]
    fn a_device_listed_twice_is_kept_once() {
        assert_eq!(
            merge_detected(&[on(radeon()), off(radeon())], &[]),
            [on(radeon())]
        );
    }

    #[test]
    fn the_processor_cannot_be_switched_off() {
        let order = DeviceOrder::saved(vec![off(DeviceKey::cpu())]);
        assert_eq!(order.effective(&[]), [on(DeviceKey::cpu())]);
    }

    #[test]
    fn the_first_device_leads_and_is_switched_on_for_the_run() {
        let order = DeviceOrder::saved(vec![on(rtx()), off(radeon()), on(DeviceKey::cpu())])
            .with_first(Some(radeon()));
        assert_eq!(
            order.effective(&[rtx(), radeon(), DeviceKey::cpu()]),
            [on(radeon()), on(rtx()), on(DeviceKey::cpu())]
        );
    }

    #[test]
    fn cpu_only_is_decided_without_the_detected_devices() {
        assert!(DeviceOrder::cpu_only().is_cpu_only());
        assert!(!DeviceOrder::automatic().is_cpu_only());
        assert!(
            DeviceOrder::automatic()
                .with_defaults(DeviceDefaults {
                    gpus_enabled: false,
                    backends_first: Vec::new()
                })
                .is_cpu_only()
        );
        assert!(!DeviceOrder::saved(vec![on(rtx()), on(DeviceKey::cpu())]).is_cpu_only());
        assert!(DeviceOrder::saved(vec![off(rtx()), on(DeviceKey::cpu())]).is_cpu_only());
        assert!(DeviceOrder::saved(vec![on(DeviceKey::cpu()), on(rtx())]).is_cpu_only());
        assert!(
            !DeviceOrder::cpu_only()
                .with_first(Some(rtx()))
                .is_cpu_only()
        );
    }
}
