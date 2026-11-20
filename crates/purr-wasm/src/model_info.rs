//! Conversion of pinned web models into the platform-neutral model description.

use crate::model_cache;
use purr_common::{
    platform::ModelInfo,
    web_model::registry::{WebModel, WebModelSpec},
};
use std::path::PathBuf;

/// The model as people see it, sized and marked installed for the variant `spec` that would run.
#[must_use]
pub fn model_info(model: &WebModel, spec: &WebModelSpec, is_installed: bool) -> ModelInfo {
    let baseline = model.baseline();
    let mut info = ModelInfo::new(
        spec.id.clone(),
        baseline.display_name(),
        baseline.description.clone(),
        spec.total_size_bytes(),
        false,
        baseline.descriptor(),
    )
    .with_download_url(spec.repo_url());
    if is_installed {
        info.mark_installed(PathBuf::from(model_cache::location(spec)));
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::{
        language::support::LanguageSupport,
        model::{
            descriptor::Weights,
            engine::EngineId,
            family::{ModelFamily, WhisperSize},
            format::ModelFormat,
            profile::{DetectionReport, HintEffect, InputLimit, ModelProfile, Speakers, Timing},
        },
        web_model::{attributes::OnnxDtype, registry::WebModelRegistry},
    };

    fn info(id: &str, installed: bool) -> Result<ModelInfo, Box<dyn std::error::Error>> {
        let registry = WebModelRegistry::embedded()?;
        let model = registry.get(id)?;
        Ok(model_info(model, model.baseline(), installed))
    }

    #[test]
    fn the_size_is_that_of_the_variant_shown() -> Result<(), Box<dyn std::error::Error>> {
        let registry = WebModelRegistry::embedded()?;
        let model = registry.get("base")?;
        for spec in model.variants() {
            let shown = model_info(model, spec, false);
            assert_eq!(shown.size_bytes, spec.total_size_bytes());
            assert_eq!(shown.descriptor, model.baseline().descriptor());
        }
        Ok(())
    }

    #[test]
    fn english_only_models_know_only_english() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            info("base.en", false)?.descriptor.profile.languages,
            LanguageSupport::english_only()
        );
        assert_eq!(
            info("base", false)?.descriptor.profile.languages,
            LanguageSupport::Any
        );
        Ok(())
    }

    #[test]
    fn every_browser_model_is_described_by_the_browser_profile()
    -> Result<(), Box<dyn std::error::Error>> {
        let registry = WebModelRegistry::embedded()?;
        for model in registry.models() {
            let spec = model.baseline();
            let descriptor = model_info(model, spec, false).descriptor;
            assert_eq!(descriptor.engine, EngineId::Browser, "{}", spec.id);
            assert_eq!(descriptor.format, ModelFormat::Onnx, "{}", spec.id);
            assert_eq!(
                descriptor.profile,
                ModelProfile {
                    languages: spec.languages.clone(),
                    language_hint: HintEffect::Steers,
                    detection: DetectionReport::Reported,
                    timing: Timing::Segment,
                    word_scores: false,
                    speakers: Speakers::None,
                    input: InputLimit::Unbounded,
                },
                "{}",
                spec.id
            );
        }
        Ok(())
    }

    #[test]
    fn the_weights_name_the_encoder_and_decoder_dtypes() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            info("base", false)?.descriptor.weights,
            Weights::Onnx {
                encoder: OnnxDtype::Fp32,
                decoder: OnnxDtype::Q4
            }
        );
        assert_eq!(
            info("base-q4", false)?.descriptor.weights,
            Weights::Onnx {
                encoder: OnnxDtype::Q4,
                decoder: OnnxDtype::Q4
            }
        );
        assert_eq!(
            info("base", false)?.descriptor.family,
            ModelFamily::Whisper(WhisperSize::Base)
        );
        Ok(())
    }

    #[test]
    fn only_the_turbo_models_need_shader_f16() -> Result<(), Box<dyn std::error::Error>> {
        for id in ["large-v3-turbo", "large-v3-turbo-q4f16"] {
            assert!(info(id, false)?.descriptor.weights.requires_shader_f16());
        }
        assert!(
            !info("small", false)?
                .descriptor
                .weights
                .requires_shader_f16()
        );
        Ok(())
    }

    #[test]
    fn only_the_fp16_turbo_model_is_webgpu_only() -> Result<(), Box<dyn std::error::Error>> {
        assert!(
            !info("large-v3-turbo", false)?
                .descriptor
                .weights
                .runs_on_cpu()
        );
        assert!(
            info("large-v3-turbo-q4f16", false)?
                .descriptor
                .weights
                .runs_on_cpu()
        );
        assert!(info("tiny", false)?.descriptor.weights.runs_on_cpu());
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
