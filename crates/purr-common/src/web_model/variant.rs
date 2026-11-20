//! The ways a browser model is offered: with half-precision weights for GPUs that have
//! `shader-f16`, and with full-precision ones for everything else.

use serde_with::{DeserializeFromStr, SerializeDisplay};
use strum::{Display, EnumString};

/// Which of a model's weight sets is installed and run.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Display,
    EnumString,
    SerializeDisplay,
    DeserializeFromStr,
)]
pub enum VariantKind {
    /// Needs the WebGPU `shader-f16` feature.
    #[strum(serialize = "fp16")]
    Fp16,
    #[strum(serialize = "fp32")]
    Fp32,
}

/// How the embedding table shared by a decoder's lookup and output head is stored.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Hash,
    Display,
    EnumString,
    SerializeDisplay,
    DeserializeFromStr,
)]
pub enum TableStorage {
    #[default]
    #[strum(serialize = "fp32")]
    Fp32,
    #[strum(serialize = "fp16")]
    Fp16,
}

impl TableStorage {
    #[must_use]
    pub const fn is_fp32(&self) -> bool {
        match self {
            Self::Fp32 => true,
            Self::Fp16 => false,
        }
    }
}

/// Whether the WebGPU adapter offers the `shader-f16` feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderF16 {
    Supported,
    Missing,
}
