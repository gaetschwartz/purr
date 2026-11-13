//! Which hardware transcription runs on: what the user may choose and what a machine reports.

use crate::{devices::DeviceKey, platform::DeviceInfo};
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

/// The backend setting of version 2 settings files: let Purr pick, or one backend by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AccelSelection {
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
}

/// Why a backend cannot be used on this machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendUnavailable {
    /// This copy of Purr was built without it.
    NotBuilt,
    /// Built in, but no usable device was found.
    NoDevice,
    /// Built in, but the runtime library it loads at start cannot be found.
    RuntimeLibraryMissing {
        library: String,
        /// The directories looked in before the system's own library search.
        #[serde(default)]
        searched: Vec<String>,
    },
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
    AvxVnni,
    Avx2,
    Bmi2,
    Fma,
    F16c,
    Avx512F,
    Avx512Bw,
    Avx512Vl,
    Avx512Vbmi,
    Avx512Vnni,
    Avx512Bf16,
    AmxInt8,
}

impl CpuFeature {
    /// Every instruction set, in the order they are listed.
    pub const ALL: [Self; 13] = [
        Self::Avx,
        Self::AvxVnni,
        Self::Avx2,
        Self::Bmi2,
        Self::Fma,
        Self::F16c,
        Self::Avx512F,
        Self::Avx512Bw,
        Self::Avx512Vl,
        Self::Avx512Vbmi,
        Self::Avx512Vnni,
        Self::Avx512Bf16,
        Self::AmxInt8,
    ];

    /// The name chip makers use.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Avx => "AVX",
            Self::AvxVnni => "AVX-VNNI",
            Self::Avx2 => "AVX2",
            Self::Bmi2 => "BMI2",
            Self::Fma => "FMA",
            Self::F16c => "F16C",
            Self::Avx512F => "AVX512F",
            Self::Avx512Bw => "AVX512BW",
            Self::Avx512Vl => "AVX512VL",
            Self::Avx512Vbmi => "AVX512VBMI",
            Self::Avx512Vnni => "AVX512VNNI",
            Self::Avx512Bf16 => "AVX512BF16",
            Self::AmxInt8 => "AMX-INT8",
        }
    }

    /// Whether the ggml build says if it was compiled with this instruction set.
    #[must_use]
    pub const fn reported_by_build(self) -> bool {
        match self {
            Self::Avx512Bw | Self::Avx512Vl => false,
            Self::Avx
            | Self::AvxVnni
            | Self::Avx2
            | Self::Bmi2
            | Self::Fma
            | Self::F16c
            | Self::Avx512F
            | Self::Avx512Vbmi
            | Self::Avx512Vnni
            | Self::Avx512Bf16
            | Self::AmxInt8 => true,
        }
    }

    /// Whether the running processor can be asked for this instruction set.
    #[must_use]
    pub const fn probed_at_runtime(self) -> bool {
        match self {
            Self::AmxInt8 => false,
            Self::Avx
            | Self::AvxVnni
            | Self::Avx2
            | Self::Bmi2
            | Self::Fma
            | Self::F16c
            | Self::Avx512F
            | Self::Avx512Bw
            | Self::Avx512Vl
            | Self::Avx512Vbmi
            | Self::Avx512Vnni
            | Self::Avx512Bf16 => true,
        }
    }

    /// The labels of `features`, comma separated.
    #[must_use]
    pub fn join_labels(features: &[Self]) -> String {
        features
            .iter()
            .map(|feature| feature.label())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// What the processor offers and what this build was compiled to use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuReport {
    /// The processor's marketing name, when the system gives one.
    pub name: Option<String>,
    /// Threads the processor runs at once, when the system says.
    pub logical_threads: Option<u32>,
    /// Instruction sets the binary was compiled with.
    pub built_with: Vec<CpuFeature>,
    /// Instruction sets the running processor reports; `None` where this platform cannot be asked.
    pub detected: Option<Vec<CpuFeature>>,
}

impl CpuReport {
    /// Instruction sets the binary needs that the processor lacks.
    #[must_use]
    pub fn missing_on_cpu(&self) -> Vec<CpuFeature> {
        let Some(detected) = &self.detected else {
            return Vec::new();
        };
        self.built_with
            .iter()
            .copied()
            .filter(|feature| feature.probed_at_runtime() && !detected.contains(feature))
            .collect()
    }

    /// Instruction sets the processor has that the binary was not compiled with.
    #[must_use]
    pub fn unused_by_build(&self) -> Vec<CpuFeature> {
        self.detected
            .iter()
            .flatten()
            .copied()
            .filter(|feature| feature.reported_by_build() && !self.built_with.contains(feature))
            .collect()
    }
}

/// Why the last transcription did not run on a backend it considered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PassedReason {
    /// The backend could not be used at all.
    Unavailable(BackendUnavailable),
    /// The backend looked usable but failed while loading the model.
    Failed(String),
}

/// A device the last transcription passed over before it found one that worked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassedOver {
    pub key: DeviceKey,
    pub reason: PassedReason,
}

/// What the last transcription of this session ran on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LastRun {
    pub backend: AccelBackend,
    /// The device or processor name, as the system reports it.
    pub device: Option<String>,
    pub passed_over: Vec<PassedOver>,
}

/// The backend and hardware a saved transcription ran on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineUsed {
    pub backend: AccelBackend,
    /// The device or processor name, when the system gave one.
    pub hardware: Option<String>,
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

    fn report(built_with: &[CpuFeature], detected: Option<&[CpuFeature]>) -> CpuReport {
        CpuReport {
            name: None,
            logical_threads: None,
            built_with: built_with.to_vec(),
            detected: detected.map(<[CpuFeature]>::to_vec),
        }
    }

    #[test]
    fn every_instruction_set_is_listed_once() {
        let mut seen = Vec::new();
        for feature in CpuFeature::ALL {
            assert!(!seen.contains(&feature));
            seen.push(feature);
        }
        assert_eq!(seen.len(), 13);
    }

    #[test]
    fn labels_are_joined_in_the_given_order() {
        assert_eq!(
            CpuFeature::join_labels(&[CpuFeature::Avx2, CpuFeature::Avx512F]),
            "AVX2, AVX512F"
        );
        assert_eq!(CpuFeature::join_labels(&[]), "");
    }

    #[test]
    fn a_build_matching_the_processor_has_no_mismatch() {
        let both = [CpuFeature::Avx2, CpuFeature::Fma];
        let report = report(&both, Some(&both));
        assert_eq!(report.missing_on_cpu(), []);
        assert_eq!(report.unused_by_build(), []);
    }

    #[test]
    fn instruction_sets_the_binary_needs_but_the_processor_lacks_are_missing() {
        let report = report(
            &[CpuFeature::Avx2, CpuFeature::Avx512F, CpuFeature::AmxInt8],
            Some(&[CpuFeature::Avx2]),
        );
        assert_eq!(report.missing_on_cpu(), [CpuFeature::Avx512F]);
    }

    #[test]
    fn instruction_sets_the_build_cannot_report_are_never_unused() {
        let report = report(
            &[CpuFeature::Avx2],
            Some(&[
                CpuFeature::Avx2,
                CpuFeature::Avx512F,
                CpuFeature::Avx512Bw,
                CpuFeature::Avx512Vl,
            ]),
        );
        assert_eq!(report.unused_by_build(), [CpuFeature::Avx512F]);
    }

    #[test]
    fn a_platform_that_cannot_be_probed_reports_no_mismatch() {
        let report = report(&[CpuFeature::Avx2], None);
        assert_eq!(report.missing_on_cpu(), []);
        assert_eq!(report.unused_by_build(), []);
    }

    #[test]
    fn cpu_is_the_last_resort() {
        assert_eq!(AccelBackend::PRIORITY.last(), Some(&AccelBackend::Cpu));
    }
}
