//! What the shared pipeline produces with a Qwen3-ASR model: text without times, a language the
//! model names, and windows that stay within what it takes.
//!
//! Needs `Qwen3-ASR-0.6B-Q8_0.gguf` and the Silero model in Purr's model cache, and the libraries
//! transcribe.cpp is built into. Run with
//! `cargo nextest run -p purr-engine-transcribe --test qwen3 --run-ignored only --no-capture`.

use futures::StreamExt;
use purr_common::{
    devices::DeviceOrder,
    language::tag::LanguageTag,
    model::{engine::EngineId, profile::InputLimit},
    platform::ChunkKind,
};
use purr_core::{
    AudioStream, ModelManager,
    audio::AudioProcessor,
    config::{ModelRef, RunConfig, Windowing},
    engine::{Engine, open::open_on_best_device},
    pipeline::{
        events::{StreamEvent, StreamingChunk},
        run::{Input, transcribe},
    },
};
use purr_engine_transcribe::engine::TranscribeEngine;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

const MODEL: &str = "Qwen3-ASR-0.6B-Q8_0.gguf";
const THREADS: usize = 4;
const SAMPLE_RATE: usize = 16_000;
const EXCERPT_SECONDS: usize = 60;
const JFK: &str = "And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.";

fn samples_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples")
}

fn config(vad: Option<PathBuf>) -> RunConfig {
    RunConfig::new()
        .with_devices(DeviceOrder::cpu_only())
        .with_threads(THREADS)
        .with_vad_model(vad)
}

fn engines() -> Vec<Arc<dyn Engine>> {
    vec![Arc::new(TranscribeEngine)]
}

async fn committed(
    model: &Path,
    config: RunConfig,
    samples: &[f32],
) -> TestResult<Vec<StreamingChunk>> {
    let mut stream = transcribe(
        Input::Audio(AudioStream::from_samples(samples)),
        config,
        &ModelRef::new(model, EngineId::TranscribeCpp),
        &engines(),
    )
    .await?;
    let mut chunks = Vec::new();
    while let Some(event) = stream.next().await {
        if let StreamEvent::Chunk(chunk) = event?
            && chunk.kind == ChunkKind::Committed
        {
            println!("[{:6.2} - {:6.2}] {}", chunk.start, chunk.end, chunk.text);
            chunks.push(chunk);
        }
    }
    Ok(chunks)
}

fn text(chunks: &[StreamingChunk]) -> String {
    chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs the Qwen3-ASR and Silero models in the model cache"]
async fn qwen3_asr_transcribes_through_the_shared_pipeline() -> TestResult {
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

    let opened = open_on_best_device(&TranscribeEngine, &model, &DeviceOrder::cpu_only(), THREADS)?;
    let limits = opened.decoder.limits();
    println!("limits: {limits:?}");
    assert!(matches!(limits.input, InputLimit::UpTo(_)));
    drop(opened);

    let jfk = AudioProcessor::load_audio(samples_dir().join("jfk.wav"))
        .await?
        .samples;
    let chunks = committed(&model, config(Some(vad.clone())), &jfk).await?;
    assert_eq!(text(&chunks), JFK);
    assert!(
        chunks
            .iter()
            .all(|chunk| chunk.language == Some(LanguageTag::english()))
    );

    let mut a13 = AudioProcessor::load_audio(samples_dir().join("a13.mp3"))
        .await?
        .samples;
    a13.truncate(EXCERPT_SECONDS * SAMPLE_RATE);
    for config in [
        config(Some(vad)),
        config(None),
        config(None).with_windowing(Windowing::Whole),
    ] {
        let chunks = committed(&model, config, &a13).await?;
        assert!(!text(&chunks).is_empty());
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.start <= chunk.end && chunk.end <= EXCERPT_SECONDS as f64)
        );
        assert!(chunks.windows(2).all(|pair| pair[0].end <= pair[1].start));
    }
    Ok(())
}
