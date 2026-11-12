//! Conversion of pinned web models into the platform-neutral model description.

use crate::model_cache;
use purr_common::platform::{ModelInfo, ModelMetadata, model_flag};
use purr_common::web_model::registry::WebModelSpec;
use std::path::PathBuf;

#[must_use]
pub fn model_info(spec: &WebModelSpec, is_installed: bool) -> ModelInfo {
    let metadata = ModelMetadata::new(
        "onnx".to_owned(),
        "whisper".to_owned(),
        if spec.english_only() {
            "english".to_owned()
        } else {
            "multilingual".to_owned()
        },
        spec.revision.clone(),
    )
    .with_download_url(spec.repo_url())
    .with_quantization(format!("{}/{}", spec.encoder_dtype, spec.decoder_dtype))
    .with_platform_attribute("webgpu_compatible".to_owned(), "true".to_owned())
    .with_platform_attribute(
        model_flag::REQUIRES_SHADER_F16.to_owned(),
        spec.requires_shader_f16().to_string(),
    )
    .with_platform_attribute(
        model_flag::ENGLISH_ONLY.to_owned(),
        spec.english_only().to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::web_model::registry::WebModelRegistry;

    fn info(id: &str, installed: bool) -> Result<ModelInfo, Box<dyn std::error::Error>> {
        let registry = WebModelRegistry::embedded()?;
        Ok(model_info(registry.get(id)?, installed))
    }

    #[test]
    fn english_only_models_are_flagged_and_labelled() -> Result<(), Box<dyn std::error::Error>> {
        let english = info("base.en", false)?;
        assert!(english.metadata.has_flag(model_flag::ENGLISH_ONLY));
        assert_eq!(english.metadata.language_support, "english");
        let multilingual = info("base", false)?;
        assert!(!multilingual.metadata.has_flag(model_flag::ENGLISH_ONLY));
        assert_eq!(multilingual.metadata.language_support, "multilingual");
        Ok(())
    }

    #[test]
    fn the_quantization_names_the_encoder_and_decoder_dtypes()
    -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            info("base", false)?.metadata.quantization.as_deref(),
            Some("fp32/q4")
        );
        assert_eq!(
            info("base-q4", false)?.metadata.quantization.as_deref(),
            Some("q4/q4")
        );
        Ok(())
    }

    #[test]
    fn only_the_turbo_models_need_shader_f16() -> Result<(), Box<dyn std::error::Error>> {
        for id in ["large-v3-turbo", "large-v3-turbo-q4f16"] {
            assert!(
                info(id, false)?
                    .metadata
                    .has_flag(model_flag::REQUIRES_SHADER_F16)
            );
        }
        assert!(
            !info("small", false)?
                .metadata
                .has_flag(model_flag::REQUIRES_SHADER_F16)
        );
        Ok(())
    }

    #[test]
    fn installed_state_follows_the_argument() -> Result<(), Box<dyn std::error::Error>> {
        assert!(info("tiny", true)?.is_installed);
        assert!(!info("tiny", false)?.is_installed);
        Ok(())
    }

    #[test]
    fn id_is_the_registry_id_and_name_is_readable() -> Result<(), Box<dyn std::error::Error>> {
        let tiny = info("tiny.en", false)?;
        assert_eq!(tiny.id, "tiny.en");
        assert_eq!(tiny.name, "Whisper Tiny (English)");
        Ok(())
    }
}
