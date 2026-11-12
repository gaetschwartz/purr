//! Where a browser transcription runs: the GPU through WebGPU, or the CPU through WebAssembly.

use serde::{Deserialize, Serialize};

/// The execution provider a browser transcription ran on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrowserEngine {
    WebGpu,
    Cpu(CpuReason),
}

/// Why a browser transcription runs on the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpuReason {
    /// The settings ask for the CPU.
    Chosen,
    /// The browser offers no WebGPU adapter.
    NoWebGpu,
    /// The GPU transcribed a known recording wrongly.
    WrongOutput,
    /// WebGPU failed while loading or running the model.
    WebGpuFailed,
}

impl CpuReason {
    /// Whether the next run, whatever its model, is likely to go to the CPU for the same reason.
    #[must_use]
    pub const fn concerns_the_browser(self) -> bool {
        match self {
            Self::NoWebGpu | Self::WrongOutput | Self::WebGpuFailed => true,
            Self::Chosen => false,
        }
    }

    /// One short sentence for a run that went to the CPU on its own; `None` when it was asked for.
    #[must_use]
    pub const fn notice(self) -> Option<&'static str> {
        match self {
            Self::Chosen => None,
            Self::NoWebGpu => {
                Some("This browser has no WebGPU, so Purr uses the CPU. It's slower.")
            }
            Self::WrongOutput => {
                Some("This browser's GPU gave wrong results, so Purr uses the CPU. It's slower.")
            }
            Self::WebGpuFailed => {
                Some("WebGPU failed in this browser, so Purr uses the CPU. It's slower.")
            }
        }
    }
}
