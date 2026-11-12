use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::backend::AccelBackend;

/// Why a backend could not be considered at all.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
pub enum UnavailableReason {
    #[error("support for it was not compiled into this build")]
    NotBuilt,
    #[error("support for it is compiled in but the runtime library {library} cannot be loaded")]
    RuntimeLibraryMissing {
        library: String,
        /// The directories looked in before the system's own library search.
        #[serde(default)]
        searched: Vec<PathBuf>,
    },
    #[error("support for it is compiled in but no usable device was detected")]
    NoDevice,
    #[error("support for it is not compiled in, or no usable device was detected")]
    NotBuiltOrNoDevice,
}

impl UnavailableReason {
    /// The usual fix for `backend` being unavailable for this reason, when there is one.
    #[must_use]
    pub const fn remedy(&self, backend: AccelBackend) -> Option<&'static str> {
        match self {
            Self::RuntimeLibraryMissing { .. } => backend.runtime_library_hint(),
            Self::NotBuilt | Self::NoDevice | Self::NotBuiltOrNoDevice => None,
        }
    }
}

/// Why a backend that looked available failed while loading the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
pub enum AttemptFailure {
    #[error("device {device} failed to initialise")]
    DeviceInit { device: String },
    #[error("no CoreML encoder found at {}", path.display())]
    CoreMlModelMissing { path: PathBuf },
    #[error("whisper could not create its context")]
    ContextInit,
    #[error("whisper could not create its state")]
    StateInit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttemptOutcome {
    Unavailable(UnavailableReason),
    Failed(AttemptFailure),
}

impl std::fmt::Display for AttemptOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(reason) => reason.fmt(f),
            Self::Failed(failure) => failure.fmt(f),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccelAttempt {
    pub backend: AccelBackend,
    pub outcome: AttemptOutcome,
}

/// What acceleration a transcription ended up on, and what was passed over to get there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccelReport {
    pub selected: AccelBackend,
    pub device: Option<String>,
    /// Backends tried or considered before `selected`, in priority order.
    pub attempts: Vec<AccelAttempt>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn missing_library() -> UnavailableReason {
        UnavailableReason::RuntimeLibraryMissing {
            library: "vulkan-1.dll".to_owned(),
            searched: Vec::new(),
        }
    }

    #[test]
    fn a_missing_vulkan_library_has_a_remedy() {
        assert_eq!(
            missing_library().remedy(AccelBackend::Vulkan),
            Some(
                "install your GPU's Vulkan driver (on Linux, Intel and AMD use Mesa's Vulkan driver)"
            )
        );
    }

    #[test]
    fn a_missing_library_of_another_backend_has_no_remedy() {
        assert_eq!(missing_library().remedy(AccelBackend::Cuda), None);
    }

    #[test]
    fn other_reasons_have_no_remedy() {
        for reason in [
            UnavailableReason::NotBuilt,
            UnavailableReason::NoDevice,
            UnavailableReason::NotBuiltOrNoDevice,
        ] {
            assert_eq!(reason.remedy(AccelBackend::Vulkan), None);
        }
    }
}
