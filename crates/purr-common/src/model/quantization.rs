use super::precision::Precision;
use serde_with::{DeserializeFromStr, SerializeDisplay};
use strum::{Display, EnumString};

/// How the weights of a GGML or GGUF model file are stored.
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
pub enum Quantization {
    #[strum(serialize = "f32")]
    F32,
    #[strum(serialize = "f16")]
    F16,
    #[strum(serialize = "q8_0")]
    Q8_0,
    #[strum(serialize = "q6_k")]
    Q6K,
    #[strum(serialize = "q5_1")]
    Q5_1,
    #[strum(serialize = "q5_0")]
    Q5_0,
    #[strum(serialize = "q5_k_m")]
    Q5KM,
    #[strum(serialize = "q4_k_m")]
    Q4KM,
}

impl Quantization {
    /// The name the format is known by, such as `Q4_K_M`.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q8_0 => "Q8_0",
            Self::Q6K => "Q6_K",
            Self::Q5_1 => "Q5_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5KM => "Q5_K_M",
            Self::Q4KM => "Q4_K_M",
        }
    }

    #[must_use]
    pub const fn precision(self) -> Precision {
        match self {
            Self::F32 | Self::F16 => Precision::Full,
            Self::Q8_0 => Precision::Quantized { bits: 8 },
            Self::Q6K => Precision::Quantized { bits: 6 },
            Self::Q5_1 | Self::Q5_0 | Self::Q5KM => Precision::Quantized { bits: 5 },
            Self::Q4KM => Precision::Quantized { bits: 4 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_quantization_has_its_precision() {
        let precisions: Vec<_> = [
            Quantization::F32,
            Quantization::F16,
            Quantization::Q8_0,
            Quantization::Q6K,
            Quantization::Q5_1,
            Quantization::Q5_0,
            Quantization::Q5KM,
            Quantization::Q4KM,
        ]
        .into_iter()
        .map(Quantization::precision)
        .collect();
        assert_eq!(
            precisions,
            [
                Precision::Full,
                Precision::Full,
                Precision::Quantized { bits: 8 },
                Precision::Quantized { bits: 6 },
                Precision::Quantized { bits: 5 },
                Precision::Quantized { bits: 5 },
                Precision::Quantized { bits: 5 },
                Precision::Quantized { bits: 4 },
            ]
        );
    }

    #[test]
    fn a_label_is_the_stored_name_in_capitals() {
        for quantization in [
            Quantization::F32,
            Quantization::F16,
            Quantization::Q8_0,
            Quantization::Q6K,
            Quantization::Q5_1,
            Quantization::Q5_0,
            Quantization::Q5KM,
            Quantization::Q4KM,
        ] {
            assert_eq!(
                quantization.label().to_lowercase().parse(),
                Ok(quantization)
            );
        }
    }

    #[test]
    fn the_names_transcribe_cpp_ships_are_read_in_lower_case() {
        for (name, quantization) in [
            ("F32", Quantization::F32),
            ("F16", Quantization::F16),
            ("Q8_0", Quantization::Q8_0),
            ("Q6_K", Quantization::Q6K),
            ("Q5_K_M", Quantization::Q5KM),
            ("Q4_K_M", Quantization::Q4KM),
            ("Q5_0", Quantization::Q5_0),
            ("Q5_1", Quantization::Q5_1),
        ] {
            assert_eq!(name.to_lowercase().parse(), Ok(quantization));
        }
    }
}
