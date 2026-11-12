use crate::{
    hooks::use_models::use_models,
    platform::{IS_WEB, Platform as _, get_platform},
    transcribe::{engine_label::PlannedEngine, session::use_transcribe_session},
};
use dioxus::prelude::*;
use purr_common::accel::AccelPolicy;

/// Where the next browser run of `model_id` under `policy` goes, worked out again when either
/// changes, when the GPU is found, and after every run; pending until the model list has loaded.
pub fn use_engine_plan(model_id: Option<String>, policy: AccelPolicy) -> PlannedEngine {
    let session = use_transcribe_session();
    let models = use_models();
    let plan = use_resource(use_reactive!(|(model_id, policy)| async move {
        let _last_run = session.browser_engine();
        let _gpu = models.gpu.read().found();
        if !IS_WEB {
            return PlannedEngine::Unknown;
        }
        let Some(id) = model_id else {
            return PlannedEngine::Pending;
        };
        match get_platform().plan_browser_engine(&id, policy).await {
            Ok(plan) => PlannedEngine::Known(plan),
            Err(error) => {
                tracing::warn!("Could not tell where {id} will run: {error}");
                PlannedEngine::Unknown
            }
        }
    }));
    plan.read().unwrap_or(PlannedEngine::Pending)
}
