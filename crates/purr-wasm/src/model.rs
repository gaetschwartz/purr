//! Browser model registry backed by the pinned web manifest.

use crate::error::WebResult;
use crate::model_cache;
use purr_common::web_model::registry::{WebModel, WebModelRegistry, WebModelSpec};
use std::sync::Arc;

/// Lists the pinned browser models and reports which ones transformers.js has cached.
pub struct WebModelManager {
    registry: Arc<WebModelRegistry>,
}

impl WebModelManager {
    /// Manager over the manifest embedded in this build.
    pub fn new() -> WebResult<Self> {
        Ok(Self::from_registry(WebModelRegistry::embedded()?))
    }

    #[must_use]
    pub const fn from_registry(registry: Arc<WebModelRegistry>) -> Self {
        Self { registry }
    }

    #[must_use]
    pub fn models(&self) -> &[Arc<WebModel>] {
        self.registry.models()
    }

    pub fn model(&self, model_id: &str) -> WebResult<&WebModel> {
        Ok(self.registry.get(model_id)?)
    }

    #[must_use]
    pub fn default_model(&self) -> &WebModel {
        self.registry.default_model()
    }

    /// Whether every file of the variant is in the browser cache.
    pub async fn is_installed(&self, spec: &WebModelSpec) -> WebResult<bool> {
        model_cache::is_installed(spec).await
    }

    /// Evicts the files of every variant of the model from the browser cache, keeping those an
    /// installed sibling still loads.
    pub async fn remove_model(&self, model_id: &str) -> WebResult<()> {
        let model = self.model(model_id)?;
        let mut kept = Vec::new();
        for sibling in self.registry.siblings(model) {
            for spec in sibling.variants() {
                if model_cache::is_installed(spec).await? {
                    kept.push(spec);
                }
            }
        }
        for spec in model.variants() {
            model_cache::remove(spec, &spec.files_not_loaded_by(&kept)).await?;
        }
        Ok(())
    }
}
