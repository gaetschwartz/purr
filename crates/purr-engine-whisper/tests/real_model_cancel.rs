//! Cancelling a decode stops whisper.cpp inside it.
//!
//! The decode is raced against a timer, so the audio has to take far longer to decode than the
//! timer takes to fire.

mod common;

use common::WhisperModel;
use purr_common::model::profile::Timing;
use purr_core::{
    AudioProcessor,
    engine::{
        cancel::CancelToken,
        decoder::{DecodeRequest, LanguageRequest},
        hypothesis::Completion,
        open::open_on_best_device,
    },
};
use purr_engine_whisper::engine::WhisperEngine;
use std::{path::Path, time::Duration};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cancelled_decode_stops_early_with_what_it_finished()
-> Result<(), Box<dyn std::error::Error>> {
    let model = common::model_path(WhisperModel::Base).await?;
    let samples = AudioProcessor::load_audio(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples/a13.mp3"),
    )
    .await?
    .samples;
    let config = common::run_config();
    let mut opened = open_on_best_device(
        &WhisperEngine,
        &model,
        &config.devices,
        config.thread_count(),
    )?;

    let cancel = CancelToken::new();
    let trigger = cancel.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(1));
        trigger.cancel();
    });
    let decoded = opened.decoder.decode(DecodeRequest {
        pcm: &samples,
        language: &LanguageRequest::Use("en".parse()?),
        timing: Timing::Segment,
        cancel: &cancel,
    })?;
    assert_eq!(decoded.completion, Completion::Cancelled);
    Ok(())
}
