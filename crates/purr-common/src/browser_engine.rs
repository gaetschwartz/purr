//! Where a browser transcription runs: the GPU through WebGPU, or the CPU through WebAssembly.

use crate::accel::{AccelBackend, AccelPolicy};
use serde::{Deserialize, Serialize};

/// The execution provider a browser transcription runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrowserEngine {
    WebGpu,
    Cpu(CpuReason),
}

/// Why a browser transcription does not run on WebGPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpuReason {
    /// The settings ask for the CPU.
    Chosen,
    /// The browser offers no WebGPU adapter.
    NoWebGpu,
    /// The model needs the `shader-f16` feature and the adapter lacks it.
    NoShaderF16,
    /// WebGPU failed while loading or running the model.
    WebGpuFailed,
}

impl BrowserEngine {
    /// The engine the settings ask for: the CPU when they say so, WebGPU otherwise.
    #[must_use]
    pub const fn requested(policy: AccelPolicy) -> Self {
        match policy {
            AccelPolicy::Forced {
                backend: AccelBackend::Cpu,
                ..
            } => Self::Cpu(CpuReason::Chosen),
            AccelPolicy::Auto
            | AccelPolicy::Forced {
                backend:
                    AccelBackend::Cuda
                    | AccelBackend::Hip
                    | AccelBackend::CoreMl
                    | AccelBackend::Metal
                    | AccelBackend::Sycl
                    | AccelBackend::Vulkan,
                ..
            } => Self::WebGpu,
        }
    }
}

impl CpuReason {
    /// One short sentence for a run that went to the CPU on its own; `None` when it was asked for.
    #[must_use]
    pub const fn notice(self) -> Option<&'static str> {
        match self {
            Self::Chosen => None,
            Self::NoWebGpu => {
                Some("This browser has no WebGPU, so Purr uses the CPU. It's slower.")
            }
            Self::NoShaderF16 => {
                Some("This GPU lacks shader-f16, so this model runs on the CPU. It's slower.")
            }
            Self::WebGpuFailed => {
                Some("WebGPU failed in this browser, so Purr uses the CPU. It's slower.")
            }
        }
    }

    /// One short sentence on why a model that only runs on WebGPU cannot run here.
    #[must_use]
    pub const fn gpu_only_refusal(self) -> &'static str {
        match self {
            Self::Chosen => "This model can't run on the CPU, which Settings chose.",
            Self::NoWebGpu => "This model needs WebGPU, which this browser lacks.",
            Self::NoShaderF16 => "This model needs shader-f16, which this GPU lacks.",
            Self::WebGpuFailed => {
                "WebGPU failed in this browser, and this model can't run on the CPU."
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::Settings;

    #[test]
    fn a_fresh_profile_asks_for_webgpu() {
        assert_eq!(
            BrowserEngine::requested(Settings::default().performance.accel_policy()),
            BrowserEngine::WebGpu
        );
    }

    #[test]
    fn only_the_processor_asks_for_the_cpu() {
        assert_eq!(
            BrowserEngine::requested(AccelPolicy::CPU_ONLY),
            BrowserEngine::Cpu(CpuReason::Chosen)
        );
        assert_eq!(
            BrowserEngine::requested(AccelPolicy::Forced {
                backend: AccelBackend::Cpu,
                allow_fallback: true,
            }),
            BrowserEngine::Cpu(CpuReason::Chosen)
        );
        assert_eq!(
            BrowserEngine::requested(AccelPolicy::Forced {
                backend: AccelBackend::Vulkan,
                allow_fallback: false,
            }),
            BrowserEngine::WebGpu
        );
    }

    #[test]
    fn only_a_run_that_moved_on_its_own_explains_itself() {
        assert_eq!(CpuReason::Chosen.notice(), None);
        for reason in [
            CpuReason::NoWebGpu,
            CpuReason::NoShaderF16,
            CpuReason::WebGpuFailed,
        ] {
            assert!(reason.notice().is_some(), "{reason:?}");
        }
    }
}
