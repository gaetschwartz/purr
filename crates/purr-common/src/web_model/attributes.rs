//! Typed attributes describing a browser Whisper model.

use serde_with::{DeserializeFromStr, SerializeDisplay};
use strum::{Display, EnumString};

/// Model size class. Declaration order is ascending size.
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
pub enum WebModelFamily {
    #[strum(serialize = "tiny")]
    Tiny,
    #[strum(serialize = "base")]
    Base,
    #[strum(serialize = "small")]
    Small,
    #[strum(serialize = "large-v3-turbo")]
    LargeV3Turbo,
}

impl WebModelFamily {
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Tiny => "Whisper Tiny",
            Self::Base => "Whisper Base",
            Self::Small => "Whisper Small",
            Self::LargeV3Turbo => "Whisper Large v3 Turbo",
        }
    }
}

/// ONNX weight precision as named by transformers.js dtype options.
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
pub enum OnnxDtype {
    #[strum(serialize = "fp32")]
    Fp32,
    #[strum(serialize = "fp16")]
    Fp16,
    #[strum(serialize = "q8")]
    Q8,
    #[strum(serialize = "int8")]
    Int8,
    #[strum(serialize = "uint8")]
    Uint8,
    #[strum(serialize = "q4")]
    Q4,
    #[strum(serialize = "bnb4")]
    Bnb4,
    #[strum(serialize = "q4f16")]
    Q4F16,
}

impl OnnxDtype {
    /// File name suffix transformers.js appends to the model file for this dtype.
    #[must_use]
    pub const fn file_suffix(self) -> &'static str {
        match self {
            Self::Fp32 => "",
            Self::Fp16 => "_fp16",
            Self::Q8 => "_quantized",
            Self::Int8 => "_int8",
            Self::Uint8 => "_uint8",
            Self::Q4 => "_q4",
            Self::Bnb4 => "_bnb4",
            Self::Q4F16 => "_q4f16",
        }
    }

    /// Whether running this dtype on WebGPU needs the `shader-f16` adapter feature.
    #[must_use]
    pub const fn requires_shader_f16(self) -> bool {
        match self {
            Self::Fp16 | Self::Q4F16 => true,
            Self::Fp32 | Self::Q8 | Self::Int8 | Self::Uint8 | Self::Q4 | Self::Bnb4 => false,
        }
    }
}
