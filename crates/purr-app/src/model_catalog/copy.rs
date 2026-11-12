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
    pub body: String,
    /// Whether the cause is the connection, which colours the message as a warning.
    pub is_connection: bool,
}

/// What to tell the user about a failed download.
#[must_use]
pub fn failure(failure: &DownloadFailure) -> FailureCopy {
    match failure.kind {
        ModelFailureKind::Network => FailureCopy {
            chip: "Offline",
            title: "Couldn't reach the download server",
            body: "Check your connection and try again.".to_owned(),
            is_connection: true,
        },
        ModelFailureKind::Integrity => FailureCopy {
            chip: "Failed",
            title: "The download didn't match its checksum",
            body: format!(
                "The files were removed so nothing broken is kept. Try again. {}",
                failure.message
            ),
            is_connection: false,
        },
        ModelFailureKind::Other => FailureCopy {
            chip: "Failed",
            title: "The download failed",
            body: failure.message.clone(),
            is_connection: false,
        },
    }
}

/// What a failed integrity check means, and what to do about it.
#[must_use]
pub fn damage(damage: &Damage) -> String {
    let what = match damage.problem {
        IntegrityProblem::Missing => format!("{} is missing.", damage.file),
        IntegrityProblem::SizeMismatch => format!(
            "{} has the wrong size, probably from an interrupted write.",
            damage.file
        ),
        IntegrityProblem::ChecksumMismatch => format!(
            "{} doesn't match its published checksum, probably from an interrupted write.",
            damage.file
        ),
    };
    format!("{what} Download the model again to repair it.")
}

/// The chip and hint of a model that cannot run here.
#[must_use]
pub const fn unavailable(reason: UnavailableReason) -> (&'static str, &'static str) {
    match reason {
        UnavailableReason::NeedsShaderF16 => ("Needs shader-f16", "This GPU lacks shader-f16."),
        UnavailableReason::NeedsWebGpu => ("Needs WebGPU", "This browser has no WebGPU."),
        UnavailableReason::CpuChosen => (
            "Needs WebGPU",
            "Too large for the CPU, which Settings chose.",
        ),
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
    fn an_unexplained_failure_shows_the_platform_message() {
        assert_eq!(failure(&failed(ModelFailureKind::Other)).body, "detail");
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
