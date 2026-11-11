use purr_common::web_model::registry::WebModelRegistry;
use purr_wasm::WorkerModel;

pub fn default_worker_model() -> WorkerModel {
    let registry = WebModelRegistry::embedded().expect("the embedded web manifest is valid");
    WorkerModel::from(registry.default_model())
}
