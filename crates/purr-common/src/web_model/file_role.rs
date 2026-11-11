//! The files transformers.js loads for a Whisper pipeline.

use crate::web_model::attributes::OnnxDtype;

/// Purpose of a file within a model repository.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ModelFileRole {
    Config,
    GenerationConfig,
    PreprocessorConfig,
    Tokenizer,
    TokenizerConfig,
    Encoder,
    DecoderMerged,
}

impl ModelFileRole {
    /// Every role a model needs, in manifest order.
    pub const ALL: [Self; 7] = [
        Self::Config,
        Self::GenerationConfig,
        Self::PreprocessorConfig,
        Self::Tokenizer,
        Self::TokenizerConfig,
        Self::Encoder,
        Self::DecoderMerged,
    ];

    /// Repository path of this file for the given session dtypes.
    #[must_use]
    pub fn path(self, encoder: OnnxDtype, decoder: OnnxDtype) -> String {
        match self {
            Self::Config => "config.json".to_owned(),
            Self::GenerationConfig => "generation_config.json".to_owned(),
            Self::PreprocessorConfig => "preprocessor_config.json".to_owned(),
            Self::Tokenizer => "tokenizer.json".to_owned(),
            Self::TokenizerConfig => "tokenizer_config.json".to_owned(),
            Self::Encoder => format!("onnx/encoder_model{}.onnx", encoder.file_suffix()),
            Self::DecoderMerged => {
                format!("onnx/decoder_model_merged{}.onnx", decoder.file_suffix())
            }
        }
    }
}
