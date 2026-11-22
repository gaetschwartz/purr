//! One loaded model serves several inputs and gives what a fresh load gives.

mod common;

use common::WhisperModel;
use pretty_assertions::assert_eq;
use purr_common::model::engine::EngineId;
use purr_core::{
    config::ModelRef,
    engine::Engine,
    pipeline::{
        model::LoadedModel,
        result::collect,
        run::{Input, transcribe},
    },
};
use purr_engine_whisper::engine::WhisperEngine;
use std::{path::PathBuf, sync::Arc};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_model_loaded_model_matches_fresh_load() -> TestResult {
    let path = common::model_path(WhisperModel::Base).await?;
    let sample = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/jfk.wav");
    let engines: Vec<Arc<dyn Engine>> = vec![Arc::new(WhisperEngine)];
    let model = ModelRef::new(path, EngineId::WhisperCpp);
    let run = common::run_config();

    let fresh =
        collect(transcribe(Input::File(sample.clone()), run.clone(), &model, &engines).await?)
            .await?
            .text;

    let mut loaded = LoadedModel::load(&model, &run, &engines).await?;
    let first = collect(loaded.transcribe(Input::File(sample.clone()), &run).await?)
        .await?
        .text;
    let second = collect(loaded.transcribe(Input::File(sample), &run).await?)
        .await?
        .text;

    assert_eq!(first, fresh);
    assert_eq!(second, fresh);
    Ok(())
}
