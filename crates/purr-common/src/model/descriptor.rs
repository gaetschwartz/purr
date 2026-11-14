use super::{
    engine::EngineId, family::ModelFamily, format::ModelFormat, precision::Precision,
    profile::ModelProfile, quantization::Quantization,
};
use crate::web_model::attributes::{OnnxDtype, pair_precision};
use serde::{Deserialize, Serialize};

/// How the weights of a model are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Weights {
    Quantized(Quantization),
    Onnx {
        encoder: OnnxDtype,
        decoder: OnnxDtype,
    },
}

impl Weights {
    #[must_use]
    pub fn precision(self) -> Precision {
        match self {
            Self::Quantized(quantization) => quantization.precision(),
            Self::Onnx { encoder, decoder } => pair_precision(encoder, decoder),
        }
    }

    /// Whether running the weights on WebGPU needs the `shader-f16` adapter feature.
    #[must_use]
    pub const fn requires_shader_f16(self) -> bool {
        match self {
            Self::Quantized(_) => false,
            Self::Onnx { encoder, decoder } => {
                encoder.requires_shader_f16() || decoder.requires_shader_f16()
            }
        }
    }

    /// Whether the weights run on the processor, not only on a graphics device.
    #[must_use]
    pub const fn runs_on_cpu(self) -> bool {
        match self {
            Self::Quantized(_) => true,
            Self::Onnx { encoder, decoder } => encoder.runs_on_cpu() && decoder.runs_on_cpu(),
        }
    }
}

/// What a platform knows about a model beyond its name and size.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub engine: EngineId,
    pub family: ModelFamily,
    pub format: ModelFormat,
    pub weights: Weights,
    pub profile: ModelProfile,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ggml_weights_need_no_graphics_feature() {
        let weights = Weights::Quantized(Quantization::Q5_1);
        assert!(!weights.requires_shader_f16());
        assert!(weights.runs_on_cpu());
        assert_eq!(weights.precision(), Precision::Quantized { bits: 5 });
    }

    #[test]
    fn onnx_weights_follow_their_dtypes() {
        let turbo = Weights::Onnx {
            encoder: OnnxDtype::Fp16,
            decoder: OnnxDtype::Q4,
        };
        assert!(turbo.requires_shader_f16());
        assert!(!turbo.runs_on_cpu());
        assert_eq!(turbo.precision(), Precision::Mixed);
        let base = Weights::Onnx {
            encoder: OnnxDtype::Fp32,
            decoder: OnnxDtype::Q4,
        };
        assert!(!base.requires_shader_f16());
        assert!(base.runs_on_cpu());
    }
}
