//! A stream fed at the pace of speech, as a recording device would feed it.
//!
//! Tentative text is produced while the pipeline waits for audio, so the feed has to take real
//! time: this test lasts as long as the sample it plays.

mod common;

use common::WhisperModel;
use futures::StreamExt;
use purr_common::{language::tag::LanguageTag, model::engine::EngineId, platform::ChunkKind};
use purr_core::{
    AudioChunk, AudioProcessor, AudioStream, SourceLength,
    config::{ModelRef, TentativeText},
    engine::Engine,
    pipeline::{
        events::StreamEvent,
        run::{Input, transcribe},
    },
};
use purr_engine_whisper::engine::WhisperEngine;
use std::{sync::Arc, time::Duration};
use tokio::sync::mpsc;

const HALF_SECOND: usize = 8_000;

/// Feeds `samples` in half seconds at the pace of speech, from a thread of its own.
fn paced(samples: Vec<f32>) -> AudioStream {
    let (tx, rx) = mpsc::channel(8);
    std::thread::spawn(move || {
        let parts = samples.chunks(HALF_SECOND).count();
        for (i, part) in samples.chunks(HALF_SECOND).enumerate() {
            std::thread::sleep(Duration::from_millis(500));
            let chunk = AudioChunk::new(part.to_vec(), i, i as f32 * 0.5, i + 1 == parts);
            if tx.blocking_send(Ok(chunk)).is_err() {
                return;
            }
        }
    });
    AudioStream::new(rx, SourceLength::Unknown)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tentative_text_shows_early_and_leaves_the_final_text_alone()
-> Result<(), Box<dyn std::error::Error>> {
    let model = common::model_path(WhisperModel::Base).await?;
    let audio = AudioProcessor::load_audio(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples/jfk.wav"),
    )
    .await?;
    let engines: Vec<Arc<dyn Engine>> = vec![Arc::new(WhisperEngine)];
    let mut final_texts = Vec::new();
    for tentative in [
        TentativeText::Off,
        TentativeText::WhileWaiting { every_seconds: 2.0 },
    ] {
        let config = common::run_config()
            .with_language(LanguageTag::english())
            .with_tentative(tentative);
        let mut stream = transcribe(
            Input::Audio(paced(audio.samples.clone())),
            config,
            &ModelRef::new(&model, EngineId::WhisperCpp),
            &engines,
        )
        .await?;
        let mut tentative_chunks = 0;
        let mut finals = Vec::new();
        while let Some(event) = stream.next().await {
            let StreamEvent::Chunk(chunk) = event? else {
                continue;
            };
            match chunk.kind {
                ChunkKind::Tentative => tentative_chunks += 1,
                ChunkKind::Committed => finals.push(chunk.text),
            }
        }
        assert_eq!(
            tentative_chunks > 0,
            tentative != TentativeText::Off,
            "{tentative:?}"
        );
        final_texts.push(finals);
    }
    assert_eq!(final_texts[0], final_texts[1]);
    Ok(())
}
