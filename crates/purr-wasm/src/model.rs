//! Browser model registry backed by the pinned web manifest.

use crate::error::WebResult;
use crate::model_cache;
use purr_common::web_model::registry::{WebModelRegistry, WebModelSpec};
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
    pub fn models(&self) -> &[Arc<WebModelSpec>] {
        self.registry.models()
    }

    pub fn model(&self, model_id: &str) -> WebResult<&WebModelSpec> {
        Ok(self.registry.get(model_id)?)
    }

    #[must_use]
    pub fn default_model(&self) -> &WebModelSpec {
        self.registry.default_model()
    }

    /// Whether every file of the model is in the browser cache.
    pub async fn is_model_installed(&self, model_id: &str) -> WebResult<bool> {
        model_cache::is_installed(self.model(model_id)?).await
    }

    /// Evicts the model's files from the browser cache, keeping those an installed sibling
    /// variant still loads.
    pub async fn remove_model(&self, model_id: &str) -> WebResult<()> {
        let spec = self.model(model_id)?;
        let mut kept = Vec::new();
        for sibling in self.registry.siblings(spec) {
            if model_cache::is_installed(sibling).await? {
                kept.push(sibling);
            }
        }
        model_cache::remove(spec, &spec.files_not_loaded_by(&kept)).await
    }
}
