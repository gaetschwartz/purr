//! Where a browser transcription runs: the GPU through WebGPU, or the CPU through WebAssembly.

use serde::{Deserialize, Serialize};

/// The execution provider a browser transcription ran on.
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
    /// The GPU transcribed a known recording wrongly.
    WrongOutput,
    /// WebGPU failed while loading or running the model.
    WebGpuFailed,
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
            Self::WrongOutput => {
                Some("This browser's GPU gave wrong results, so Purr uses the CPU. It's slower.")
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
            Self::WrongOutput => {
                "This browser's GPU gives wrong results, and this model can't run on the CPU."
            }
            Self::WebGpuFailed => {
                "WebGPU failed in this browser, and this model can't run on the CPU."
            }
        }
    }
}

/// Where the next browser run of a model goes, as far as it can be told without running it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnginePlan {
    /// WebGPU, checked recently on this adapter, browser and runtime.
    WebGpu,
    /// WebGPU, unless the GPU check that runs first finds it unusable.
    WebGpuAfterCheck,
    Cpu(CpuReason),
    /// The model would have to run on the CPU for this reason, and it cannot.
    CannotRun(CpuReason),
}

impl From<BrowserEngine> for EnginePlan {
    fn from(engine: BrowserEngine) -> Self {
        match engine {
            BrowserEngine::WebGpu => Self::WebGpu,
            BrowserEngine::Cpu(reason) => Self::Cpu(reason),
        }
    }
}
