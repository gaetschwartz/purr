use std::path::{Path, PathBuf};

/// Where whisper.cpp looks for the CoreML encoder of a model file: the model name without its
/// extension and without a trailing quantization tag, plus `-encoder.mlmodelc`.
#[must_use]
pub fn encoder_path(model: &Path) -> PathBuf {
    let stem = model
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let base = match stem.rsplit_once('-') {
        Some((head, tag))
            if tag.len() == 4 && tag.starts_with('q') && tag.as_bytes().get(2) == Some(&b'_') =>
        {
            head
        }
        Some(_) | None => stem.as_str(),
    };
    model.with_file_name(format!("{base}-encoder.mlmodelc"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_model_maps_to_its_encoder() {
        assert_eq!(
            encoder_path(Path::new("models/ggml-base.bin")),
            Path::new("models/ggml-base-encoder.mlmodelc")
        );
    }

    #[test]
    fn quantization_tag_is_dropped() {
        assert_eq!(
            encoder_path(Path::new("models/ggml-base-q5_1.bin")),
            Path::new("models/ggml-base-encoder.mlmodelc")
        );
    }

    #[test]
    fn other_dashed_suffixes_are_kept() {
        assert_eq!(
            encoder_path(Path::new("ggml-large-v3-turbo.bin")),
            Path::new("ggml-large-v3-turbo-encoder.mlmodelc")
        );
    }
}
