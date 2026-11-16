//! What the shared pipeline produces with a Qwen3-ASR model, compared with recorded goldens: text
//! without times and a language the model names.
//!
//! `PURR_GOLDEN=update` records the goldens again.

mod common;

use common::{Mode, Sample, TestResult, TranscribeModel};
use purr_common::model::profile::InputLimit;
use purr_core::engine::open::open_on_best_device;
use purr_engine_transcribe::engine::TranscribeEngine;
use rstest::rstest;

#[rstest]
#[case::jfk_stream(Sample::Jfk, Mode::Stream, 0.0)]
#[case::jfk_whole(Sample::Jfk, Mode::Whole, 0.0)]
#[case::mm1_stream(Sample::Mm1, Mode::Stream, 0.06)]
#[case::mm1_whole(Sample::Mm1, Mode::Whole, 0.06)]
#[case::a13_stream(Sample::A13Excerpt, Mode::Stream, 0.07)]
#[case::a13_stream_novad(Sample::A13Excerpt, Mode::StreamWithoutVad, 0.14)]
#[case::a13_whole(Sample::A13Excerpt, Mode::Whole, 0.14)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn qwen3_output_matches_the_golden(
    #[case] sample: Sample,
    #[case] mode: Mode,
    #[case] max_word_error_rate: f64,
) -> TestResult {
    common::assert_golden(TranscribeModel::Qwen3, sample, mode, max_word_error_rate).await
}

#[tokio::test]
async fn qwen3_takes_a_bounded_input() -> TestResult {
    let model = common::model_path(TranscribeModel::Qwen3).await?;
    let config = common::run_config();
    let opened = open_on_best_device(
        &TranscribeEngine,
        &model,
        &config.devices,
        config.thread_count(),
    )?;
    assert!(matches!(opened.decoder.limits().input, InputLimit::UpTo(_)));
    Ok(())
}
