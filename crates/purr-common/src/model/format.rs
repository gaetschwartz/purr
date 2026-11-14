use serde_with::{DeserializeFromStr, SerializeDisplay};
use strum::{Display, EnumString};

/// How a model's weights are stored in its file.
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
pub enum ModelFormat {
    #[strum(serialize = "ggml-bin")]
    GgmlBin,
    #[strum(serialize = "gguf")]
    Gguf,
    #[strum(serialize = "onnx")]
    Onnx,
}
