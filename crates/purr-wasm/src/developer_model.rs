//! Where a developer model lives in the browser cache and how it is described to the pipeline.
//!
//! Its files sit under a repository name of their own, so they never share a cache key with a
//! catalog model, whatever Hugging Face repository they came from.

use crate::error::{WebError, WebResult};
use crate::local::run_local;
use crate::model_cache::open_existing;
use crate::protocol::ExternalData;
use purr_common::developer::model::{BrowserModel, DeveloperModel, DeveloperModelId};
use purr_common::language::support::LanguageSupport;
use purr_common::web_model::file_role::ModelFileRole;
use purr_common::web_model::manifest::file_url;
use purr_common::web_model::registry::{WebModel, WebModelSpec};
use purr_common::web_model::variant::VariantKind;
use wasm_bindgen_futures::JsFuture;

const REPO_NAMESPACE: &str = "purr-developer";
const REVISION: &str = "main";

#[must_use]
pub fn repo(id: &DeveloperModelId) -> String {
    format!("{REPO_NAMESPACE}/{}", id.as_str())
}

#[must_use]
pub fn cached_url(id: &DeveloperModelId, path: &str) -> String {
    file_url(&repo(id), REVISION, path)
}

#[must_use]
pub fn data_path(onnx_path: &str) -> String {
    format!("{onnx_path}_data")
}

/// The path of each role's file for the model's dtypes.
#[must_use]
pub fn path(role: ModelFileRole, model: &BrowserModel) -> String {
    role.path(model.encoder_dtype, model.decoder_dtype)
}

/// The cache keys that must exist for the model to load: every file but the optional generation
/// config, and the weights file of a converted model.
#[must_use]
pub fn required_urls(id: &DeveloperModelId, model: &BrowserModel) -> Vec<String> {
    let mut urls = Vec::new();
    for role in ModelFileRole::ALL {
        if role == ModelFileRole::GenerationConfig {
            continue;
        }
        let file = path(role, model);
        if role.is_onnx() && model.conversion.table().is_some() {
            urls.push(cached_url(id, &data_path(&file)));
        }
        urls.push(cached_url(id, &file));
    }
    urls
}

async fn has_all(urls: Vec<String>) -> WebResult<bool> {
    let Some(cache) = open_existing().await? else {
        return Ok(false);
    };
    for url in &urls {
        let hit = JsFuture::from(cache.match_with_str(url))
            .await
            .map_err(|e| WebError::web_api("Cache.match", e))?;
        if hit.is_undefined() {
            return Ok(false);
        }
    }
    Ok(true)
}

pub async fn is_installed(id: &DeveloperModelId, model: &BrowserModel) -> WebResult<bool> {
    let urls = required_urls(id, model);
    run_local(move || has_all(urls)).await
}

/// Which of the two ONNX files have weights stored beside them.
pub async fn external_data(id: &DeveloperModelId, model: &BrowserModel) -> WebResult<ExternalData> {
    let encoder = cached_url(id, &data_path(&path(ModelFileRole::Encoder, model)));
    let decoder = cached_url(id, &data_path(&path(ModelFileRole::DecoderMerged, model)));
    run_local(move || async move {
        Ok(ExternalData {
            encoder: has_all(vec![encoder]).await?,
            decoder: has_all(vec![decoder]).await?,
        })
    })
    .await
}

/// Drops every file the model could have in the cache.
pub async fn remove(id: &DeveloperModelId, model: &BrowserModel) -> WebResult<()> {
    let mut urls = Vec::new();
    for role in ModelFileRole::ALL {
        let file = path(role, model);
        if role.is_onnx() {
            urls.push(cached_url(id, &data_path(&file)));
        }
        urls.push(cached_url(id, &file));
    }
    run_local(move || async move {
        let Some(cache) = open_existing().await? else {
            return Ok(());
        };
        for url in &urls {
            JsFuture::from(cache.delete_with_str(url))
                .await
                .map_err(|e| WebError::web_api("Cache.delete", e))?;
        }
        Ok(())
    })
    .await
}

/// The pipeline's view of a developer model: one variant with an empty file list, so nothing
/// is checked against a manifest. The variant follows what the dtypes and the table need, and
/// there is no other to switch to.
#[must_use]
#[allow(
    clippy::needless_update,
    reason = "a catalog spec may gain fields this does not set"
)]
pub fn web_model(model: &DeveloperModel, browser: &BrowserModel, base: &WebModelSpec) -> WebModel {
    let table = browser.conversion.table().unwrap_or_default();
    let needs_half = browser.encoder_dtype.requires_shader_f16()
        || browser.decoder_dtype.requires_shader_f16()
        || !table.is_fp32();
    WebModel::single(WebModelSpec {
        id: model.id.as_str().to_owned(),
        repo: repo(&model.id),
        revision: REVISION.to_owned(),
        family: model.family,
        languages: if browser.english_only {
            LanguageSupport::english_only()
        } else {
            LanguageSupport::Any
        },
        variant: if needs_half {
            VariantKind::Fp16
        } else {
            VariantKind::Fp32
        },
        encoder_dtype: browser.encoder_dtype,
        decoder_dtype: browser.decoder_dtype,
        table,
        description: format!("Developer model {}", model.name),
        files: Vec::new(),
        ..base.clone()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::developer::model::{BrowserOrigin, Conversion};
    use purr_common::web_model::attributes::OnnxDtype;
    use purr_common::web_model::variant::TableStorage;

    fn browser(conversion: Conversion) -> BrowserModel {
        BrowserModel {
            origin: BrowserOrigin::LocalFiles,
            encoder_dtype: OnnxDtype::Fp32,
            decoder_dtype: OnnxDtype::Q4,
            conversion,
            english_only: false,
        }
    }

    #[test]
    fn files_live_under_a_namespace_no_catalog_repository_uses() {
        let id = DeveloperModelId::from_unique("0123456789ab");
        assert_eq!(
            cached_url(&id, "config.json"),
            "https://huggingface.co/purr-developer/dev-0123456789ab/resolve/main/config.json"
        );
    }

    #[test]
    fn a_converted_model_needs_its_weights_files_and_never_the_optional_generation_config() {
        let id = DeveloperModelId::from_unique("0123456789ab");
        let urls = required_urls(
            &id,
            &browser(Conversion::Streaming {
                table: TableStorage::Fp32,
            }),
        );
        let names: Vec<&str> = urls
            .iter()
            .filter_map(|url| url.rsplit_once("/main/").map(|(_, name)| name))
            .collect();
        assert_eq!(
            names,
            [
                "config.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "onnx/encoder_model.onnx_data",
                "onnx/encoder_model.onnx",
                "onnx/decoder_model_merged_q4.onnx_data",
                "onnx/decoder_model_merged_q4.onnx",
            ]
        );
    }

    #[test]
    fn a_model_used_as_it_is_needs_no_weights_files() {
        let id = DeveloperModelId::from_unique("0123456789ab");
        let urls = required_urls(&id, &browser(Conversion::AsIs));
        assert_eq!(urls.len(), 6);
    }

    fn variant_of(
        encoder: OnnxDtype,
        conversion: Conversion,
    ) -> Result<VariantKind, Box<dyn std::error::Error>> {
        let registry = purr_common::web_model::registry::WebModelRegistry::embedded()?;
        let model = DeveloperModel {
            id: DeveloperModelId::from_unique("0123456789ab"),
            name: "tiny".to_owned(),
            family: purr_common::model::family::ModelFamily::Whisper(
                purr_common::model::family::WhisperSize::Tiny,
            ),
            source: purr_common::developer::model::DeveloperSource::Browser(browser(
                Conversion::AsIs,
            )),
        };
        let mut chosen = browser(conversion);
        chosen.encoder_dtype = encoder;
        let web = web_model(&model, &chosen, registry.default_model().baseline());
        assert_eq!(web.variants().count(), 1);
        Ok(web.baseline().variant)
    }

    #[test]
    fn a_developer_model_is_the_one_variant_its_dtypes_and_table_call_for()
    -> Result<(), Box<dyn std::error::Error>> {
        let full = Conversion::Streaming {
            table: TableStorage::Fp32,
        };
        let half = Conversion::Streaming {
            table: TableStorage::Fp16,
        };
        assert_eq!(variant_of(OnnxDtype::Fp32, full)?, VariantKind::Fp32);
        assert_eq!(variant_of(OnnxDtype::Fp32, half)?, VariantKind::Fp16);
        assert_eq!(variant_of(OnnxDtype::Fp16, full)?, VariantKind::Fp16);
        assert_eq!(
            variant_of(OnnxDtype::Fp32, Conversion::AsIs)?,
            VariantKind::Fp32
        );
        Ok(())
    }
}
