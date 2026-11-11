use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::backend::AccelBackend;

/// Why a backend could not be considered at all.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
pub enum UnavailableReason {
    #[error("support for it was not compiled into this build")]
    NotBuilt,
    #[error("support for it is compiled in but the runtime library {library} cannot be loaded")]
    RuntimeLibraryMissing { library: String },
    #[error("support for it is compiled in but no usable device was detected")]
    NoDevice,
    #[error("support for it is not compiled in, or no usable device was detected")]
    NotBuiltOrNoDevice,
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
