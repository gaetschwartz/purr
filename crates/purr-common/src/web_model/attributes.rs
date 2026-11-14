//! Typed attributes describing how a browser Whisper model is stored.

use crate::model::precision::Precision;
use serde_with::{DeserializeFromStr, SerializeDisplay};
use strum::{Display, EnumString};

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
    #[must_use]
    pub const fn precision(self) -> Precision {
        match self {
            Self::Fp32 | Self::Fp16 => Precision::Full,
            Self::Q8 | Self::Int8 | Self::Uint8 => Precision::Quantized { bits: 8 },
            Self::Q4 | Self::Bnb4 | Self::Q4F16 => Precision::Quantized { bits: 4 },
        }
    }

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

    /// Whether the WebGPU execution provider can run this dtype. Dynamically quantized 8-bit
    /// weights need integer operators it lacks, and `bnb4` weights decode to garbage there.
    #[must_use]
    pub const fn runs_on_webgpu(self) -> bool {
        match self {
            Self::Fp32 | Self::Fp16 | Self::Q4 | Self::Q4F16 => true,
            Self::Q8 | Self::Int8 | Self::Uint8 | Self::Bnb4 => false,
        }
    }

    /// Whether the WebAssembly CPU runtime can load this dtype. Half-precision weights are widened
    /// to 32 bits there, which does not fit WebAssembly memory for the large encoders that use them.
    #[must_use]
    pub const fn runs_on_cpu(self) -> bool {
        match self {
            Self::Fp16 => false,
            Self::Fp32
            | Self::Q8
            | Self::Int8
            | Self::Uint8
            | Self::Q4
            | Self::Bnb4
            | Self::Q4F16 => true,
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

/// The precision of a model whose encoder and decoder are stored as `encoder` and `decoder`.
#[must_use]
pub fn pair_precision(encoder: OnnxDtype, decoder: OnnxDtype) -> Precision {
    let (encoder, decoder) = (encoder.precision(), decoder.precision());
    if encoder == decoder {
        encoder
    } else {
        Precision::Mixed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pair_is_mixed_unless_both_halves_share_a_precision() {
        assert_eq!(
            pair_precision(OnnxDtype::Fp32, OnnxDtype::Q4),
            Precision::Mixed
        );
        assert_eq!(
            pair_precision(OnnxDtype::Q4, OnnxDtype::Q4),
            Precision::Quantized { bits: 4 }
        );
        assert_eq!(
            pair_precision(OnnxDtype::Q4F16, OnnxDtype::Q4F16),
            Precision::Quantized { bits: 4 }
        );
        assert_eq!(
            pair_precision(OnnxDtype::Q4, OnnxDtype::Q8),
            Precision::Mixed
        );
        assert_eq!(
            pair_precision(OnnxDtype::Fp32, OnnxDtype::Fp16),
            Precision::Full
        );
    }
}
