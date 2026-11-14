//! Cancelling a decode stops whisper.cpp inside it.

use std::{
    path::Path,
    time::{Duration, Instant},
};

use purr_common::devices::DeviceOrder;
use purr_core::{
    AudioProcessor, ModelManager,
    engine::{
        cancel::CancelToken,
        decoder::{DecodeRequest, LanguageRequest},
        hypothesis::Completion,
        open::open_on_best_device,
    },
};
use purr_engine_whisper::engine::WhisperEngine;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs ggml-base.bin and decodes up to a minute of audio on the CPU"]
async fn a_cancelled_decode_stops_early_with_what_it_finished()
-> Result<(), Box<dyn std::error::Error>> {
    let model = ModelManager::new()?.models_dir().join("ggml-base.bin");
    if !model.exists() {
        println!("skipping: no model at {}", model.display());
        return Ok(());
    }
    let mut samples = AudioProcessor::load_audio(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples/a13.mp3"),
    )
    .await?
    .samples;
    samples.truncate(60 * 16_000);
    let mut opened = open_on_best_device(&WhisperEngine, &model, &DeviceOrder::cpu_only(), 8)?;

    let cancel = CancelToken::new();
    let trigger = cancel.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(1));
        trigger.cancel();
    });
    let started = Instant::now();
    let decoded = opened.decoder.decode(DecodeRequest {
        pcm: &samples,
        language: &LanguageRequest::Use("en".parse()?),
        timing: purr_common::model::profile::Timing::Segment,
        cancel: &cancel,
    })?;
    println!(
        "{:?} after {:?} with {} segments: {:?}",
        decoded.completion,
        started.elapsed(),
        decoded.hypothesis.segments.len(),
        decoded
            .hypothesis
            .segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>()
    );
    assert_eq!(decoded.completion, Completion::Cancelled);
    Ok(())
}
