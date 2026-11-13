//! What to tell the user when a transcription ran on another device than the first one tried.

use purr_common::{
    accel::{AccelBackend, BackendUnavailable, LastRun, PassedOver, PassedReason},
    devices::DeviceKey,
};

/// A non-blocking explanation for a run that moved on to another device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineNotice {
    pub title: String,
    pub detail: String,
}

/// Why a device was passed over, as a clause that follows "because".
#[must_use]
pub fn why_passed_over(reason: &PassedReason) -> String {
    match reason {
        PassedReason::Unavailable(BackendUnavailable::NotBuilt) => {
            "it is not part of this build".to_owned()
        }
        PassedReason::Unavailable(BackendUnavailable::NotBuiltOrNoDevice) => {
            "it is not built in, or no device was found".to_owned()
        }
        PassedReason::Unavailable(BackendUnavailable::NoDevice) => {
            "it was not found on this computer".to_owned()
        }
        PassedReason::Unavailable(BackendUnavailable::RuntimeLibraryMissing {
            library, ..
        }) => {
            format!("{library} is missing")
        }
        PassedReason::Failed(failure) => failure.clone(),
    }
}

/// A device of a backend this build lacks is not worth a word: the list outlives the build.
#[must_use]
pub const fn routinely_skipped(reason: &PassedReason) -> bool {
    matches!(
        reason,
        PassedReason::Unavailable(
            BackendUnavailable::NotBuilt | BackendUnavailable::NotBuiltOrNoDevice
        )
    )
}

/// The backend and device as one phrase, such as "Vulkan · AMD Radeon(TM) Graphics".
#[must_use]
pub fn device_phrase(backend: AccelBackend, device: Option<&str>) -> String {
    match (backend, device) {
        (AccelBackend::Cpu, _) | (_, None) => backend.label().to_owned(),
        (_, Some(device)) => format!("{} · {device}", backend.label()),
    }
}

fn key_phrase(key: &DeviceKey) -> String {
    device_phrase(
        key.backend,
        (!key.name.is_empty()).then_some(key.name.as_str()),
    )
}

fn capitalised(text: &str) -> String {
    let mut chars = text.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().chain(chars).collect()
    })
}

fn notice(skipped: &PassedOver, ran: &LastRun) -> EngineNotice {
    let used = if ran.backend == AccelBackend::Cpu {
        "the CPU".to_owned()
    } else {
        device_phrase(ran.backend, ran.device.as_deref())
    };
    EngineNotice {
        title: format!(
            "{} isn't available, so Purr is using {used}",
            key_phrase(&skipped.key)
        ),
        detail: format!(
            "{}. You can change the device order in Settings.",
            capitalised(&why_passed_over(&skipped.reason))
        ),
    }
}

/// The notice for a run that moved past a device that could not be used, if one is due.
#[must_use]
pub fn fallback_notice(ran: &LastRun) -> Option<EngineNotice> {
    ran.passed_over
        .iter()
        .find(|passed| !routinely_skipped(&passed.reason))
        .map(|skipped| notice(skipped, ran))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda() -> DeviceKey {
        DeviceKey::new(AccelBackend::Cuda, "NVIDIA GeForce RTX 2070 SUPER", 0)
    }

    fn ran(backend: AccelBackend, device: Option<&str>, passed_over: Vec<PassedOver>) -> LastRun {
        LastRun {
            backend,
            device: device.map(str::to_owned),
            passed_over,
        }
    }

    fn passed(key: DeviceKey, reason: PassedReason) -> PassedOver {
        PassedOver { key, reason }
    }

    #[test]
    fn a_device_that_was_passed_over_is_reported_with_its_reason() {
        let run = ran(
            AccelBackend::Vulkan,
            Some("AMD Radeon(TM) Graphics"),
            vec![passed(
                cuda(),
                PassedReason::Unavailable(BackendUnavailable::NoDevice),
            )],
        );
        assert_eq!(
            fallback_notice(&run),
            Some(EngineNotice {
                title: "CUDA · NVIDIA GeForce RTX 2070 SUPER isn't available, so Purr is using Vulkan · AMD Radeon(TM) Graphics"
                    .to_owned(),
                detail: "It was not found on this computer. You can change the device order in Settings."
                    .to_owned(),
            })
        );
    }

    #[test]
    fn a_run_that_used_the_first_device_needs_no_notice() {
        let run = ran(
            AccelBackend::Cuda,
            Some("NVIDIA GeForce RTX 2070 SUPER"),
            vec![],
        );
        assert_eq!(fallback_notice(&run), None);
    }

    #[test]
    fn a_device_of_a_backend_this_build_lacks_stays_quiet() {
        let run = ran(
            AccelBackend::Cpu,
            None,
            vec![
                passed(
                    DeviceKey::new(AccelBackend::Metal, "Apple M3", 0),
                    PassedReason::Unavailable(BackendUnavailable::NotBuilt),
                ),
                passed(
                    DeviceKey::new(AccelBackend::Hip, "Instinct", 0),
                    PassedReason::Unavailable(BackendUnavailable::NotBuiltOrNoDevice),
                ),
            ],
        );
        assert_eq!(fallback_notice(&run), None);
    }

    #[test]
    fn a_device_that_looked_usable_and_failed_is_reported_after_the_quiet_ones() {
        let run = ran(
            AccelBackend::Vulkan,
            Some("AMD Radeon(TM) Graphics"),
            vec![
                passed(
                    DeviceKey::new(AccelBackend::Hip, "Instinct", 0),
                    PassedReason::Unavailable(BackendUnavailable::NotBuilt),
                ),
                passed(
                    cuda(),
                    PassedReason::Failed("the driver is too old".to_owned()),
                ),
            ],
        );
        assert_eq!(
            fallback_notice(&run).map(|notice| notice.detail),
            Some("The driver is too old. You can change the device order in Settings.".to_owned())
        );
    }

    #[test]
    fn a_missing_runtime_library_ends_on_the_cpu() {
        let run = ran(
            AccelBackend::Cpu,
            None,
            vec![passed(
                cuda(),
                PassedReason::Unavailable(BackendUnavailable::RuntimeLibraryMissing {
                    library: "cudart64_12.dll".to_owned(),
                    searched: Vec::new(),
                }),
            )],
        );
        assert_eq!(
            fallback_notice(&run).map(|notice| notice.title),
            Some(
                "CUDA · NVIDIA GeForce RTX 2070 SUPER isn't available, so Purr is using the CPU"
                    .to_owned()
            )
        );
    }

    #[test]
    fn a_second_gpu_taking_over_from_the_first_of_one_backend_is_reported() {
        let first = DeviceKey::new(AccelBackend::Vulkan, "Same Card", 0);
        let run = ran(
            AccelBackend::Vulkan,
            Some("Same Card"),
            vec![passed(first, PassedReason::Failed("it failed".to_owned()))],
        );
        assert_eq!(
            fallback_notice(&run).map(|notice| notice.title),
            Some(
                "Vulkan · Same Card isn't available, so Purr is using Vulkan · Same Card"
                    .to_owned()
            )
        );
    }
}
