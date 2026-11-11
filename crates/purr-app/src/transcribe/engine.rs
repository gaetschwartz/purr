//! Where a transcription runs and what to tell the user when it ran somewhere else than asked.

use purr_common::accel::{
    AccelBackend, AccelPolicy, BackendUnavailable, LastRun, PassedOver, PassedReason,
};

/// The hardware a finished run used, with the policy it was started under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunMemory {
    pub policy: AccelPolicy,
    pub ran: LastRun,
}

/// Where the next run is expected to happen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Planned {
    /// The processor.
    Cpu,
    /// A named backend, from the settings or from the last run under the same policy.
    Backend(AccelBackend),
    /// Some graphics card, as far as device detection can tell.
    Gpu,
}

impl Planned {
    /// Where a run will happen, from the policy, else the last run made under it, else detection.
    #[must_use]
    pub fn of(policy: AccelPolicy, memory: Option<&RunMemory>, gpu_detected: bool) -> Self {
        let remembered = memory.filter(|memory| memory.policy == policy);
        match (policy, remembered) {
            (AccelPolicy::CPU_ONLY, _) => Self::Cpu,
            (_, Some(memory)) => Self::of_backend(memory.ran.backend),
            (AccelPolicy::Forced { backend, .. }, None) => Self::of_backend(backend),
            (AccelPolicy::Auto, None) if gpu_detected => Self::Gpu,
            (AccelPolicy::Auto, None) => Self::Cpu,
        }
    }

    /// Where a run actually happened.
    #[must_use]
    pub const fn of_backend(backend: AccelBackend) -> Self {
        match backend {
            AccelBackend::Cpu => Self::Cpu,
            AccelBackend::Cuda
            | AccelBackend::Hip
            | AccelBackend::CoreMl
            | AccelBackend::Metal
            | AccelBackend::Sycl
            | AccelBackend::Vulkan => Self::Backend(backend),
        }
    }

    /// The short name for a details line.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => AccelBackend::Cpu.label(),
            Self::Backend(backend) => backend.label(),
            Self::Gpu => "GPU",
        }
    }

    /// The words for "processed on this device with ...".
    #[must_use]
    pub fn sentence(self) -> String {
        match self {
            Self::Cpu => "the CPU".to_owned(),
            Self::Backend(backend) => backend.label().to_owned(),
            Self::Gpu => "your GPU".to_owned(),
        }
    }
}

/// A non-blocking explanation for a run that used other hardware than the one asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineNotice {
    pub title: String,
    pub detail: String,
}

/// Why a backend was passed over, as a clause that follows "because".
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
            "no usable device responded".to_owned()
        }
        PassedReason::Unavailable(BackendUnavailable::RuntimeLibraryMissing { library }) => {
            format!("{library} is missing")
        }
        PassedReason::Failed(failure) => failure.clone(),
    }
}

/// Skipping a backend this build lacks, or one with no device, is how automatic selection works.
#[must_use]
pub const fn routinely_skipped(reason: &PassedReason) -> bool {
    matches!(
        reason,
        PassedReason::Unavailable(
            BackendUnavailable::NotBuilt
                | BackendUnavailable::NotBuiltOrNoDevice
                | BackendUnavailable::NoDevice
        )
    )
}

fn capitalised(text: &str) -> String {
    let mut chars = text.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().chain(chars).collect()
    })
}

fn notice(skipped: &PassedOver, ran: AccelBackend) -> EngineNotice {
    let used = if ran == AccelBackend::Cpu {
        "the CPU"
    } else {
        ran.label()
    };
    EngineNotice {
        title: format!(
            "{} isn't available, so Purr is using {used}",
            skipped.backend.label()
        ),
        detail: format!(
            "{}. You can change the backend in Settings.",
            capitalised(&why_passed_over(&skipped.reason))
        ),
    }
}

/// The notice for a run started under `requested` that ended on other hardware, if one is due.
///
/// Automatic selection says nothing about backends it expects to skip; a named backend that did
/// not run is always reported.
#[must_use]
pub fn fallback_notice(requested: AccelPolicy, ran: &LastRun) -> Option<EngineNotice> {
    match requested {
        AccelPolicy::Forced { backend, .. } if backend == ran.backend => None,
        AccelPolicy::Forced { backend, .. } => {
            let skipped = ran
                .passed_over
                .iter()
                .find(|passed| passed.backend == backend)
                .cloned()
                .unwrap_or_else(|| PassedOver {
                    backend,
                    reason: PassedReason::Failed("it could not be used".to_owned()),
                });
            Some(notice(&skipped, ran.backend))
        }
        AccelPolicy::Auto => ran
            .passed_over
            .iter()
            .find(|passed| !routinely_skipped(&passed.reason))
            .map(|skipped| notice(skipped, ran.backend)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn forced(backend: AccelBackend) -> AccelPolicy {
        AccelPolicy::Forced {
            backend,
            allow_fallback: true,
        }
    }

    fn ran(backend: AccelBackend, passed_over: Vec<PassedOver>) -> LastRun {
        LastRun {
            backend,
            device: None,
            passed_over,
        }
    }

    fn passed(backend: AccelBackend, reason: PassedReason) -> PassedOver {
        PassedOver { backend, reason }
    }

    #[test]
    fn turning_the_gpu_off_means_the_cpu_whatever_was_detected() {
        assert_eq!(Planned::of(AccelPolicy::CPU_ONLY, None, true), Planned::Cpu);
        let memory = RunMemory {
            policy: AccelPolicy::Auto,
            ran: ran(AccelBackend::Cuda, vec![]),
        };
        assert_eq!(
            Planned::of(AccelPolicy::CPU_ONLY, Some(&memory), true),
            Planned::Cpu
        );
    }

    #[test]
    fn a_named_backend_is_what_the_settings_say() {
        assert_eq!(
            Planned::of(forced(AccelBackend::Vulkan), None, false),
            Planned::Backend(AccelBackend::Vulkan)
        );
    }

    #[test]
    fn a_named_backend_that_fell_back_is_remembered_as_what_ran() {
        let policy = forced(AccelBackend::Cuda);
        let memory = RunMemory {
            policy,
            ran: ran(AccelBackend::Cpu, vec![]),
        };
        assert_eq!(Planned::of(policy, Some(&memory), true), Planned::Cpu);
        assert_eq!(
            Planned::of(forced(AccelBackend::Vulkan), Some(&memory), true),
            Planned::Backend(AccelBackend::Vulkan)
        );
    }

    #[test]
    fn automatic_trusts_the_last_run_made_the_same_way() {
        let memory = RunMemory {
            policy: AccelPolicy::Auto,
            ran: ran(AccelBackend::Vulkan, vec![]),
        };
        assert_eq!(
            Planned::of(AccelPolicy::Auto, Some(&memory), false),
            Planned::Backend(AccelBackend::Vulkan)
        );
    }

    #[test]
    fn a_run_made_under_another_policy_says_nothing_about_automatic() {
        let on_cpu = RunMemory {
            policy: AccelPolicy::CPU_ONLY,
            ran: ran(AccelBackend::Cpu, vec![]),
        };
        assert_eq!(
            Planned::of(AccelPolicy::Auto, Some(&on_cpu), true),
            Planned::Gpu
        );
        assert_eq!(
            Planned::of(AccelPolicy::Auto, Some(&on_cpu), false),
            Planned::Cpu
        );
    }

    #[test]
    fn without_any_evidence_detection_decides() {
        assert_eq!(Planned::of(AccelPolicy::Auto, None, true), Planned::Gpu);
        assert_eq!(Planned::of(AccelPolicy::Auto, None, false), Planned::Cpu);
    }

    #[test]
    fn a_named_backend_that_did_not_run_is_reported_with_its_reason() {
        let run = ran(
            AccelBackend::Vulkan,
            vec![passed(
                AccelBackend::Cuda,
                PassedReason::Unavailable(BackendUnavailable::NoDevice),
            )],
        );
        assert_eq!(
            fallback_notice(forced(AccelBackend::Cuda), &run),
            Some(EngineNotice {
                title: "CUDA isn't available, so Purr is using Vulkan".to_owned(),
                detail: "No usable device responded. You can change the backend in Settings."
                    .to_owned(),
            })
        );
    }

    #[test]
    fn a_named_backend_that_ran_needs_no_notice() {
        let run = ran(AccelBackend::Cuda, vec![]);
        assert_eq!(fallback_notice(forced(AccelBackend::Cuda), &run), None);
    }

    #[test]
    fn a_named_backend_missing_from_the_report_still_gets_a_notice() {
        let run = ran(AccelBackend::Cpu, vec![]);
        assert_eq!(
            fallback_notice(forced(AccelBackend::Metal), &run),
            Some(EngineNotice {
                title: "Metal isn't available, so Purr is using the CPU".to_owned(),
                detail: "It could not be used. You can change the backend in Settings.".to_owned(),
            })
        );
    }

    #[test]
    fn automatic_stays_quiet_about_backends_it_expects_to_skip() {
        let run = ran(
            AccelBackend::Cpu,
            vec![
                passed(
                    AccelBackend::Cuda,
                    PassedReason::Unavailable(BackendUnavailable::NoDevice),
                ),
                passed(
                    AccelBackend::Metal,
                    PassedReason::Unavailable(BackendUnavailable::NotBuilt),
                ),
            ],
        );
        assert_eq!(fallback_notice(AccelPolicy::Auto, &run), None);
    }

    #[test]
    fn automatic_reports_a_backend_that_looked_usable_and_failed() {
        let run = ran(
            AccelBackend::Vulkan,
            vec![
                passed(
                    AccelBackend::Cuda,
                    PassedReason::Failed("the driver is too old".to_owned()),
                ),
                passed(
                    AccelBackend::Hip,
                    PassedReason::Unavailable(BackendUnavailable::NoDevice),
                ),
            ],
        );
        assert_eq!(
            fallback_notice(AccelPolicy::Auto, &run),
            Some(EngineNotice {
                title: "CUDA isn't available, so Purr is using Vulkan".to_owned(),
                detail: "The driver is too old. You can change the backend in Settings.".to_owned(),
            })
        );
    }

    #[test]
    fn automatic_reports_a_missing_runtime_library() {
        let run = ran(
            AccelBackend::Cpu,
            vec![passed(
                AccelBackend::Cuda,
                PassedReason::Unavailable(BackendUnavailable::RuntimeLibraryMissing {
                    library: "cudart64_12.dll".to_owned(),
                }),
            )],
        );
        assert_eq!(
            fallback_notice(AccelPolicy::Auto, &run).map(|notice| notice.title),
            Some("CUDA isn't available, so Purr is using the CPU".to_owned())
        );
    }

    #[test]
    fn choosing_the_cpu_never_needs_a_notice() {
        let run = ran(AccelBackend::Cpu, vec![]);
        assert_eq!(fallback_notice(AccelPolicy::CPU_ONLY, &run), None);
    }
}
