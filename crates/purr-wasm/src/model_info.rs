//! Conversion of pinned web models into the platform-neutral model description.

use crate::model_cache;
use purr_common::platform::{ModelInfo, ModelMetadata};
use purr_common::web_model::registry::WebModelSpec;
use std::path::PathBuf;

#[must_use]
pub fn model_info(spec: &WebModelSpec, is_installed: bool) -> ModelInfo {
    let metadata = ModelMetadata::new(
        "onnx".to_owned(),
        "whisper".to_owned(),
        "multilingual".to_owned(),
        spec.revision.clone(),
    )
    .with_download_url(spec.repo_url())
    .with_quantization(format!("{}/{}", spec.encoder_dtype, spec.decoder_dtype))
    .with_platform_attribute("webgpu_compatible".to_owned(), "true".to_owned())
    .with_platform_attribute(
        "requires_shader_f16".to_owned(),
        spec.requires_shader_f16().to_string(),
    )
    .with_platform_attribute("repo".to_owned(), spec.repo.clone())
    .with_platform_attribute("revision".to_owned(), spec.revision.clone())
    .with_platform_attribute("family".to_owned(), spec.family.to_string())
    .with_platform_attribute("encoder_dtype".to_owned(), spec.encoder_dtype.to_string())
    .with_platform_attribute("decoder_dtype".to_owned(), spec.decoder_dtype.to_string());

    let mut info = ModelInfo::new(
        spec.id.clone(),
        spec.display_name().to_owned(),
        spec.description.clone(),
        spec.total_size_bytes(),
        false,
        metadata,
    );
    if is_installed {
        info.mark_installed(PathBuf::from(model_cache::location(spec)));
    }
    info
}
