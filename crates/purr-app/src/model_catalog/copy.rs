use super::{
    activity::{Damage, DownloadFailure},
    state::UnavailableReason,
};
use purr_common::platform::{IntegrityProblem, ModelFailureKind};

/// The words for a failed download.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailureCopy {
    pub chip: &'static str,
    pub title: &'static str,
    /// The platform's own words, for a Details disclosure.
    pub detail: Option<String>,
    /// Whether the cause is the connection, which colours the message as a warning.
    pub is_connection: bool,
}

fn given(message: &str) -> Option<String> {
    (!message.trim().is_empty()).then(|| message.to_owned())
}

/// What to tell the user about a failed download.
#[must_use]
pub fn failure(failure: &DownloadFailure) -> FailureCopy {
    match failure.kind {
        ModelFailureKind::Network => FailureCopy {
            chip: "Offline",
            title: "Couldn't reach the download server",
            detail: None,
            is_connection: true,
        },
        ModelFailureKind::Integrity => FailureCopy {
            chip: "Failed",
            title: "The download didn't match its checksum",
            detail: given(&failure.message),
            is_connection: false,
        },
        ModelFailureKind::Other => FailureCopy {
            chip: "Failed",
            title: "The download failed",
            detail: given(&failure.message),
            is_connection: false,
        },
    }
}

/// Which file failed the integrity check and how.
#[must_use]
pub fn damage(damage: &Damage) -> String {
    match damage.problem {
        IntegrityProblem::Missing => format!("{} is missing", damage.file),
        IntegrityProblem::SizeMismatch => format!("{} has the wrong size", damage.file),
        IntegrityProblem::ChecksumMismatch => {
            format!("{} doesn't match its checksum", damage.file)
        }
    }
}

/// The chip and tooltip of a model that cannot run here.
#[must_use]
pub const fn unavailable(reason: UnavailableReason) -> (&'static str, &'static str) {
    match reason {
        UnavailableReason::NeedsShaderF16 => ("Needs shader-f16", "This GPU lacks shader-f16"),
        UnavailableReason::NeedsWebGpu => ("Needs WebGPU", "This browser has no WebGPU"),
        UnavailableReason::CpuChosen => ("Needs WebGPU", "Settings chose the CPU"),
        UnavailableReason::EngineNotBuilt => {
            ("Not in this build", "Its engine isn't in this build")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failed(kind: ModelFailureKind) -> DownloadFailure {
        DownloadFailure {
            kind,
            message: "detail".to_owned(),
        }
    }

    #[test]
    fn only_a_dropped_connection_is_a_connection_problem() {
        assert!(failure(&failed(ModelFailureKind::Network)).is_connection);
        assert!(!failure(&failed(ModelFailureKind::Integrity)).is_connection);
        assert!(!failure(&failed(ModelFailureKind::Other)).is_connection);
    }

    #[test]
    fn the_platform_message_is_the_detail_of_any_but_a_connection_failure() {
        assert_eq!(
            failure(&failed(ModelFailureKind::Other)).detail.as_deref(),
            Some("detail")
        );
        assert_eq!(
            failure(&failed(ModelFailureKind::Integrity))
                .detail
                .as_deref(),
            Some("detail")
        );
        assert_eq!(failure(&failed(ModelFailureKind::Network)).detail, None);
    }

    #[test]
    fn each_gpu_reason_has_its_own_hint() {
        assert_ne!(
            unavailable(UnavailableReason::NeedsShaderF16).0,
            unavailable(UnavailableReason::NeedsWebGpu).0
        );
        assert_ne!(
            unavailable(UnavailableReason::NeedsWebGpu).1,
            unavailable(UnavailableReason::CpuChosen).1
        );
    }
}
