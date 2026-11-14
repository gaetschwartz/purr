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

impl ModelFormat {
    /// The format of a file that starts with `header`, for the formats that begin with a fixed
    /// magic number: the GGUF specification's `GGUF`, and ggml's legacy magic written as a
    /// little-endian `u32` (`0x67676d6c`). ONNX files have none.
    #[must_use]
    pub fn of_header(header: &[u8]) -> Option<Self> {
        match header.get(..4)? {
            b"GGUF" => Some(Self::Gguf),
            b"lmgg" => Some(Self::GgmlBin),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_are_told_apart_by_their_magic() {
        assert_eq!(
            ModelFormat::of_header(b"GGUF\x03\0\0\0"),
            Some(ModelFormat::Gguf)
        );
        assert_eq!(
            ModelFormat::of_header(&0x6767_6d6c_u32.to_le_bytes()),
            Some(ModelFormat::GgmlBin)
        );
        assert_eq!(ModelFormat::of_header(b"\x08\x07"), None);
        assert_eq!(ModelFormat::of_header(b"RIFF...."), None);
    }
}
