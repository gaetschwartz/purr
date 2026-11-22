//! One loaded model serves several different inputs one after the other and gives what a fresh
//! load gives for each of them, so nothing carries over from one input to the next.

mod common;

use common::{Sample, TestResult, TranscribeModel};
use pretty_assertions::assert_eq;
use purr_common::model::engine::EngineId;
use purr_core::{
    AudioStream,
    config::ModelRef,
    pipeline::{
        model::LoadedModel,
        result::collect,
        run::{Input, transcribe},
    },
};
use rstest::rstest;

#[rstest]
#[case::parakeet(TranscribeModel::Parakeet)]
#[case::qwen3(TranscribeModel::Qwen3)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_model_loaded_model_matches_fresh_loads(#[case] which: TranscribeModel) -> TestResult {
    let path = common::model_path(which).await?;
    let engines = common::engines();
    let model = ModelRef::new(path, EngineId::TranscribeCpp);
    let run = common::run_config();

    let order = [Sample::Jfk, Sample::A13Excerpt, Sample::Jfk];
    let mut fresh = Vec::new();
    for sample in order {
        let samples = common::load(sample).await?;
        let transcription = transcribe(
            Input::Audio(AudioStream::from_samples(&samples)),
            run.clone(),
            &model,
            &engines,
        )
        .await?;
        fresh.push(collect(transcription).await?.text);
    }

    let mut loaded = LoadedModel::load(&model, &run, &engines).await?;
    let mut reused = Vec::new();
    for sample in order {
        let samples = common::load(sample).await?;
        let transcription = loaded
            .transcribe(Input::Audio(AudioStream::from_samples(&samples)), &run)
            .await?;
        reused.push(collect(transcription).await?.text);
    }

    assert_eq!(reused, fresh);
    Ok(())
}
