//! Which hardware transcription runs on: what the user may choose and what a machine reports.

use crate::platform::DeviceInfo;
use serde::{Deserialize, Serialize};

/// A hardware backend a transcription can run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccelBackend {
    Cuda,
    Hip,
    CoreMl,
    Metal,
    Sycl,
    Vulkan,
    Cpu,
}

impl AccelBackend {
    /// Every backend, best first; `Cpu` is the last resort.
    pub const PRIORITY: [Self; 7] = [
        Self::Cuda,
        Self::Hip,
        Self::CoreMl,
        Self::Metal,
        Self::Sycl,
        Self::Vulkan,
        Self::Cpu,
    ];

    /// The name people know the backend by.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Hip => "ROCm",
            Self::CoreMl => "CoreML",
            Self::Metal => "Metal",
            Self::Sycl => "SYCL",
            Self::Vulkan => "Vulkan",
            Self::Cpu => "CPU",
        }
    }
}

/// The backend setting: let Purr pick, or one backend by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AccelSelection {
    /// The best backend that works, from the priority list.
    #[default]
    Auto,
    Cuda,
    Hip,
    CoreMl,
    Metal,
    Sycl,
    Vulkan,
    Cpu,
}

impl AccelSelection {
    /// Every choice, in the order a menu lists them.
    pub const ALL: [Self; 8] = [
        Self::Auto,
        Self::Cuda,
        Self::Hip,
        Self::CoreMl,
        Self::Metal,
        Self::Sycl,
        Self::Vulkan,
        Self::Cpu,
    ];

    /// The backend this choice names, `None` for [`Self::Auto`].
    #[must_use]
    pub const fn backend(self) -> Option<AccelBackend> {
        match self {
            Self::Auto => None,
            Self::Cuda => Some(AccelBackend::Cuda),
            Self::Hip => Some(AccelBackend::Hip),
            Self::CoreMl => Some(AccelBackend::CoreMl),
            Self::Metal => Some(AccelBackend::Metal),
            Self::Sycl => Some(AccelBackend::Sycl),
            Self::Vulkan => Some(AccelBackend::Vulkan),
            Self::Cpu => Some(AccelBackend::Cpu),
        }
    }

    /// The choice that names `backend`.
    #[must_use]
    pub const fn of(backend: AccelBackend) -> Self {
        match backend {
            AccelBackend::Cuda => Self::Cuda,
            AccelBackend::Hip => Self::Hip,
            AccelBackend::CoreMl => Self::CoreMl,
            AccelBackend::Metal => Self::Metal,
            AccelBackend::Sycl => Self::Sycl,
            AccelBackend::Vulkan => Self::Vulkan,
            AccelBackend::Cpu => Self::Cpu,
        }
    }

    /// The name shown in the backend menu.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self.backend() {
            None => "Automatic",
            Some(backend) => backend.label(),
        }
    }
}

/// How a transcription picks its backend, after every performance setting is taken into account.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccelPolicy {
    /// Try the priority list from best to worst, moving on when a backend is unavailable.
    #[default]
    Auto,
    /// Use `backend`; without `allow_fallback` an unavailable or failing backend is an error.
    Forced {
        backend: AccelBackend,
        allow_fallback: bool,
    },
}

impl AccelPolicy {
    /// The processor only, with nothing to fall back to.
    pub const CPU_ONLY: Self = Self::Forced {
        backend: AccelBackend::Cpu,
        allow_fallback: false,
    };
}

/// Why a backend cannot be used on this machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendUnavailable {
    /// This copy of Purr was built without it.
    NotBuilt,
    /// Built in, but no usable device was found.
    NoDevice,
    /// Built in, but the runtime library it loads at start cannot be found.
    RuntimeLibraryMissing { library: String },
    /// Not built in, or no device was found; the build cannot tell which.
    NotBuiltOrNoDevice,
}

/// Whether a backend can run here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendStatus {
    Available,
    Unavailable(BackendUnavailable),
}

/// One backend, whether it can run, and the devices it would run on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendReport {
    pub backend: AccelBackend,
    pub status: BackendStatus,
    pub devices: Vec<DeviceInfo>,
}

/// A processor instruction set that speeds Whisper up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpuFeature {
    Avx,
    Avx2,
    Fma,
    F16c,
}

impl CpuFeature {
    /// The name chip makers use.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Avx => "AVX",
            Self::Avx2 => "AVX2",
            Self::Fma => "FMA",
            Self::F16c => "F16C",
        }
    }
}

/// What the processor offers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuReport {
    /// The processor's marketing name, when the system gives one.
    pub name: Option<String>,
    /// Threads the processor runs at once, when the system says.
    pub logical_threads: Option<u32>,
    pub features: Vec<CpuFeature>,
}

/// Why the last transcription did not run on a backend it considered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PassedReason {
    /// The backend could not be used at all.
    Unavailable(BackendUnavailable),
    /// The backend looked usable but failed while loading the model.
    Failed(String),
}

/// A backend the last transcription passed over before it found one that worked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassedOver {
    pub backend: AccelBackend,
    pub reason: PassedReason,
}

/// What the last transcription of this session ran on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LastRun {
    pub backend: AccelBackend,
    pub device: Option<String>,
    pub passed_over: Vec<PassedOver>,
}

/// The acceleration a machine offers, best backend first, and what the last run used.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemReport {
    pub backends: Vec<BackendReport>,
    pub cpu: CpuReport,
    pub last_run: Option<LastRun>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_backend_has_its_own_selection() {
        for backend in AccelBackend::PRIORITY {
            assert_eq!(AccelSelection::of(backend).backend(), Some(backend));
        }
        assert_eq!(AccelSelection::Auto.backend(), None);
    }

    #[test]
    fn the_menu_lists_automatic_first_then_the_priority_order() {
        let named: Vec<_> = AccelSelection::ALL
            .iter()
            .filter_map(|selection| selection.backend())
            .collect();
        assert_eq!(AccelSelection::ALL.first(), Some(&AccelSelection::Auto));
        assert_eq!(named, AccelBackend::PRIORITY);
    }

    #[test]
    fn cpu_is_the_last_resort() {
        assert_eq!(AccelBackend::PRIORITY.last(), Some(&AccelBackend::Cpu));
    }
}
