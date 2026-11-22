//! A recording is cut into the same windows however its audio arrives.

mod common;

use common::WhisperModel;
use futures::StreamExt;
use pretty_assertions::assert_eq;
use purr_common::{model::engine::EngineId, platform::ChunkKind};
use purr_core::{
    AudioChunk, AudioProcessor, AudioStream, SourceLength,
    config::ModelRef,
    engine::Engine,
    pipeline::{events::StreamEvent, model::LoadedModel, run::Input},
};
use purr_engine_whisper::engine::WhisperEngine;
use std::{sync::Arc, time::Duration};
use tokio::sync::mpsc;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

const SECOND: usize = 16_000;

/// How long to wait before each piece of audio, and how many samples it holds.
struct Arrival {
    name: &'static str,
    pieces: Vec<(usize, Duration)>,
}

impl Arrival {
    fn in_pieces(name: &'static str, each: usize, gap: Duration, total: usize) -> Self {
        Self {
            name,
            pieces: (0..total.div_ceil(each)).map(|_| (each, gap)).collect(),
        }
    }

    fn feed(&self, samples: &[f32], length: SourceLength) -> AudioStream {
        let (tx, rx) = mpsc::channel(64);
        let samples = samples.to_vec();
        let pieces = self.pieces.clone();
        std::thread::spawn(move || {
            let mut at = 0;
            for (index, (len, delay)) in pieces.into_iter().enumerate() {
                std::thread::sleep(delay);
                let end = (at + len).min(samples.len());
                let last = end == samples.len();
                let Some(part) = samples.get(at..end) else {
                    return;
                };
                let chunk = AudioChunk::new(part.to_vec(), index, at as f32 / SECOND as f32, last);
                if tx.blocking_send(Ok(chunk)).is_err() || last {
                    return;
                }
                at = end;
            }
        });
        AudioStream::new(rx, length)
    }
}

#[derive(Debug, PartialEq)]
struct Committed {
    windows: Vec<(f64, f64)>,
    text: String,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_model_window_cuts_do_not_depend_on_arrival() -> TestResult {
    let path = common::model_path(WhisperModel::Base).await?;
    let vad = common::vad_model().await?;
    let audio = AudioProcessor::load_audio(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples/jfk.wav"),
    )
    .await?;
    let total = audio.samples.len();
    let length = SourceLength::Exact(Duration::from_secs_f64(total as f64 / SECOND as f64));
    let engines: Vec<Arc<dyn Engine>> = vec![Arc::new(WhisperEngine)];
    let model = ModelRef::new(&path, EngineId::WhisperCpp);
    let run = common::run_config().with_vad_model(Some(vad));
    let mut loaded = LoadedModel::load(&model, &run, &engines).await?;

    let arrivals = [
        Arrival {
            name: "all at once",
            pieces: vec![(total, Duration::ZERO)],
        },
        Arrival {
            name: "10 s, then the rest 700 ms later",
            pieces: vec![
                (10 * SECOND, Duration::ZERO),
                (total, Duration::from_millis(700)),
            ],
        },
        Arrival::in_pieces(
            "0.5 s pieces, 100 ms apart",
            SECOND / 2,
            Duration::from_millis(100),
            total,
        ),
        Arrival::in_pieces(
            "4 s pieces, 500 ms apart",
            4 * SECOND,
            Duration::from_millis(500),
            total,
        ),
    ];

    let mut runs: Vec<(&str, Committed)> = Vec::new();
    for arrival in &arrivals {
        let mut stream = loaded
            .transcribe(Input::Audio(arrival.feed(&audio.samples, length)), &run)
            .await?;
        let mut windows = Vec::new();
        let mut texts = Vec::new();
        while let Some(event) = stream.next().await {
            if let StreamEvent::Chunk(chunk) = event?
                && chunk.kind == ChunkKind::Committed
                && !chunk.text.is_empty()
            {
                windows.push((chunk.start, chunk.end));
                texts.push(chunk.text);
            }
        }
        runs.push((
            arrival.name,
            Committed {
                windows,
                text: texts.join(" "),
            },
        ));
    }

    let (first_name, first) = &runs[0];
    assert!(!first.windows.is_empty(), "{first_name} committed nothing");
    for (name, committed) in &runs[1..] {
        assert_eq!(committed, first, "{name} differs from {first_name}");
    }
    Ok(())
}
