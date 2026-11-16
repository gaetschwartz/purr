//! What the shared pipeline produces with a Parakeet model, compared with recorded goldens, and
//! the tentative text it carries.
//!
//! `PURR_GOLDEN=update` records the goldens again.

mod common;

use common::{Mode, Sample, TestResult, TranscribeModel};
use futures::StreamExt;
use purr_common::{model::engine::EngineId, platform::ChunkKind};
use purr_core::{
    AudioChunk, AudioStream,
    config::{ModelRef, TentativeText},
    pipeline::{
        events::StreamEvent,
        run::{Input, transcribe},
    },
};
use rstest::rstest;
use std::time::Duration;
use tokio::sync::mpsc;

const JFK: &str = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";

#[rstest]
#[case::jfk_stream(Sample::Jfk, Mode::Stream)]
#[case::jfk_whole(Sample::Jfk, Mode::Whole)]
#[case::mm1_stream(Sample::Mm1, Mode::Stream)]
#[case::mm1_whole(Sample::Mm1, Mode::Whole)]
#[case::a13_stream(Sample::A13Excerpt, Mode::Stream)]
#[case::a13_stream_novad(Sample::A13Excerpt, Mode::StreamWithoutVad)]
#[case::a13_whole(Sample::A13Excerpt, Mode::Whole)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parakeet_output_matches_the_golden(
    #[case] sample: Sample,
    #[case] mode: Mode,
) -> TestResult {
    common::assert_golden(TranscribeModel::Parakeet, sample, mode).await
}

const HALF_SECOND: usize = 8_000;

/// Feeds `samples` in half seconds at the pace of speech, from a thread of its own.
fn paced(samples: Vec<f32>) -> AudioStream {
    let (tx, rx) = mpsc::unbounded_channel();
    std::thread::spawn(move || {
        let parts = samples.chunks(HALF_SECOND).count();
        for (i, part) in samples.chunks(HALF_SECOND).enumerate() {
            std::thread::sleep(Duration::from_millis(500));
            let chunk = AudioChunk::new(part.to_vec(), i, i as f32 * 0.5, i + 1 == parts);
            if tx.send(Ok(chunk)).is_err() {
                return;
            }
        }
    });
    AudioStream::new(rx)
}

/// Lasts as long as the sample it plays: tentative text is produced while the pipeline waits for
/// audio.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tentative_text_carries_words_and_scores() -> TestResult {
    let model = common::model_path(TranscribeModel::Parakeet).await?;
    let config = common::run_config()
        .with_word_timestamps(true)
        .with_tentative(TentativeText::WhileWaiting { every_seconds: 2.0 });
    let mut stream = transcribe(
        Input::Audio(paced(common::load(Sample::Jfk).await?)),
        config,
        &ModelRef::new(&model, EngineId::TranscribeCpp),
        &common::engines(),
    )
    .await?;
    let mut tentative = Vec::new();
    let mut committed = Vec::new();
    while let Some(event) = stream.next().await {
        if let StreamEvent::Chunk(chunk) = event? {
            match chunk.kind {
                ChunkKind::Tentative => tentative.push(chunk),
                ChunkKind::Committed => committed.push(chunk),
            }
        }
    }
    for chunk in &tentative {
        assert_ne!(chunk.words.len(), 0);
        assert!(chunk.words.iter().all(|word| word.score.is_some()));
        assert_eq!(chunk.language, None);
    }
    assert!(!tentative.is_empty());
    let finals: Vec<_> = committed.iter().map(|chunk| chunk.text.as_str()).collect();
    assert_eq!(finals.join(" "), JFK);
    Ok(())
}
