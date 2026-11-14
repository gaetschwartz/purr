//! What the shared pipeline produces with a Parakeet model, on two samples.
//!
//! Needs `parakeet-tdt-0.6b-v3-Q4_K_M.gguf` and the Silero model in Purr's model cache, and the
//! libraries transcribe.cpp is built into. Run with
//! `cargo nextest run -p purr-engine-transcribe --test parakeet --run-ignored only --no-capture`.

use futures::StreamExt;
use purr_common::{devices::DeviceOrder, model::engine::EngineId, platform::ChunkKind};
use purr_core::{
    AudioStream, ModelManager,
    audio::AudioProcessor,
    config::{ModelRef, RunConfig, TentativeText, Windowing},
    engine::Engine,
    pipeline::{
        events::StreamEvent,
        result::collect,
        run::{Input, transcribe},
    },
};
use purr_engine_transcribe::engine::TranscribeEngine;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

const MODEL: &str = "parakeet-tdt-0.6b-v3-Q4_K_M.gguf";
const THREADS: usize = 4;
const SAMPLE_RATE: usize = 16_000;
const EXCERPT_SECONDS: usize = 60;
const JFK: &str = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";

#[derive(Debug, Clone, PartialEq)]
struct Line {
    start: f64,
    end: f64,
    text: String,
}

fn samples_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples")
}

async fn jfk() -> TestResult<Vec<f32>> {
    Ok(AudioProcessor::load_audio(samples_dir().join("jfk.wav"))
        .await?
        .samples)
}

async fn a13_excerpt() -> TestResult<Vec<f32>> {
    let mut samples = AudioProcessor::load_audio(samples_dir().join("a13.mp3"))
        .await?
        .samples;
    samples.truncate(EXCERPT_SECONDS * SAMPLE_RATE);
    Ok(samples)
}

fn config(vad: Option<PathBuf>) -> RunConfig {
    RunConfig::new()
        .with_devices(DeviceOrder::cpu_only())
        .with_threads(THREADS)
        .with_vad_model(vad)
        .with_tentative(TentativeText::Decoded)
        .with_word_timestamps(true)
}

fn engines() -> Vec<Arc<dyn Engine>> {
    vec![Arc::new(TranscribeEngine)]
}

async fn streamed(model: &Path, config: RunConfig, samples: &[f32]) -> TestResult<Vec<Line>> {
    let mut stream = transcribe(
        Input::Audio(AudioStream::from_samples(samples)),
        config,
        &ModelRef::new(model, EngineId::TranscribeCpp),
        &engines(),
    )
    .await?;
    let mut lines = Vec::new();
    while let Some(event) = stream.next().await {
        if let StreamEvent::Chunk(chunk) = event?
            && chunk.kind == ChunkKind::Committed
        {
            assert_eq!(chunk.language, None);
            lines.push(Line {
                start: chunk.start,
                end: chunk.end,
                text: chunk.text,
            });
        }
    }
    Ok(lines)
}

async fn whole(model: &Path, config: RunConfig, samples: &[f32]) -> TestResult<Vec<Line>> {
    let result = collect(
        transcribe(
            Input::Audio(AudioStream::from_samples(samples)),
            config.with_windowing(Windowing::Whole),
            &ModelRef::new(model, EngineId::TranscribeCpp),
            &engines(),
        )
        .await?,
    )
    .await?;
    Ok(result
        .segments
        .iter()
        .map(|segment| Line {
            start: segment.start,
            end: segment.end,
            text: segment.text.clone(),
        })
        .collect())
}

fn show(name: &str, lines: &[Line]) {
    println!("--- {name}");
    for line in lines {
        println!("[{:7.2} - {:7.2}] {}", line.start, line.end, line.text);
    }
}

fn text(lines: &[Line]) -> String {
    lines
        .iter()
        .map(|line| line.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs the Parakeet and Silero models in the model cache"]
async fn parakeet_transcribes_through_the_shared_pipeline() -> TestResult {
    let models = ModelManager::new()?;
    let model = models.models_dir().join(MODEL);
    let Some(vad) = models.vad_model_path().await? else {
        println!(
            "skipping: no Silero model in {}",
            models.models_dir().display()
        );
        return Ok(());
    };
    if !model.exists() {
        println!("skipping: no model at {}", model.display());
        return Ok(());
    }

    let samples = jfk().await?;
    let stream = streamed(&model, config(Some(vad.clone())), &samples).await?;
    let whole_file = whole(&model, config(None), &samples).await?;
    show("jfk stream", &stream);
    show("jfk whole", &whole_file);
    assert_eq!(text(&stream), JFK);
    assert_eq!(text(&whole_file), JFK);

    let samples = a13_excerpt().await?;
    let stream_vad = streamed(&model, config(Some(vad)), &samples).await?;
    let stream_plain = streamed(&model, config(None), &samples).await?;
    let whole_file = whole(&model, config(None), &samples).await?;
    show("a13 stream with VAD", &stream_vad);
    show("a13 stream without VAD", &stream_plain);
    show("a13 whole", &whole_file);
    let reference = words(&whole_file);
    for lines in [&stream_vad, &stream_plain, &whole_file] {
        assert!(is_in_order(lines, EXCERPT_SECONDS as f64));
        assert!(words(lines).abs_diff(reference) * 10 <= reference);
    }
    Ok(())
}

fn words(lines: &[Line]) -> usize {
    lines
        .iter()
        .map(|line| line.text.split_whitespace().count())
        .sum()
}

fn is_in_order(lines: &[Line], limit: f64) -> bool {
    lines
        .iter()
        .all(|line| line.start < line.end && line.end <= limit)
        && lines.windows(2).all(|pair| pair[0].end <= pair[1].start)
}
