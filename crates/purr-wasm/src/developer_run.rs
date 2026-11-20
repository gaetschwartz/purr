//! What a run needs beyond the catalog when developer mode is on.

use crate::developer_model;
use crate::error::WebResult;
use crate::protocol::WorkerDeveloper;
use purr_common::developer::model::{BrowserModel, DeveloperModel};
use purr_common::developer::session_options::SessionOptions;

/// Where the files of the model a run uses come from.
#[derive(Debug, Clone)]
pub enum ModelSource {
    Catalog,
    Developer {
        model: DeveloperModel,
        browser: BrowserModel,
    },
}

/// What the worker is told about developer mode for one request; none while the mode is off.
pub async fn worker_developer(
    source: &ModelSource,
    session_options: Option<&SessionOptions>,
) -> WebResult<Option<WorkerDeveloper>> {
    let Some(session_options) = session_options else {
        return Ok(None);
    };
    let external_data = match source {
        ModelSource::Catalog => None,
        ModelSource::Developer { model, browser } => {
            Some(developer_model::external_data(&model.id, browser).await?)
        }
    };
    Ok(Some(WorkerDeveloper {
        session_options: session_options.clone(),
        external_data,
    }))
}
