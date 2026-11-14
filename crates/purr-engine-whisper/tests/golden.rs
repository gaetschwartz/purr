//! What the Whisper pipeline produces on two samples, compared with recorded goldens.
//!
//! Needs `ggml-base.bin` (and the Silero model for the runs that use it) in Purr's model cache.
//! Run with `cargo nextest run -p purr-engine-whisper --test golden --run-ignored only`; set
//! `PURR_GOLDEN=update` to record the goldens again.

use futures::StreamExt;
use purr_common::{devices::DeviceOrder, language::tag::LanguageTag, model::engine::EngineId};
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
use purr_engine_whisper::engine::WhisperEngine;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

const THREADS: usize = 8;
const SAMPLE_RATE: usize = 16_000;
const EXCERPT_SECONDS: usize = 60;

#[derive(Debug, Clone, Copy)]
enum Sample {
    Jfk,
    /// The first minute of a13.mp3
    A13Excerpt,
}

#[derive(Debug, Clone, Copy)]
enum Mode {
    Stream { vad: bool },
    Whole,
}

struct Case {
    name: &'static str,
    sample: Sample,
    mode: Mode,
    language: Option<&'static str>,
}

const CASES: &[Case] = &[
    Case {
        name: "jfk-stream-auto",
        sample: Sample::Jfk,
        mode: Mode::Stream { vad: true },
        language: None,
    },
    Case {
        name: "jfk-stream-en",
        sample: Sample::Jfk,
        mode: Mode::Stream { vad: true },
        language: Some("en"),
    },
    Case {
        name: "jfk-whole-auto",
        sample: Sample::Jfk,
        mode: Mode::Whole,
        language: None,
    },
    Case {
        name: "jfk-whole-en",
        sample: Sample::Jfk,
        mode: Mode::Whole,
        language: Some("en"),
    },
    Case {
        name: "a13-stream-auto",
        sample: Sample::A13Excerpt,
        mode: Mode::Stream { vad: true },
        language: None,
    },
    Case {
        name: "a13-stream-novad-auto",
        sample: Sample::A13Excerpt,
        mode: Mode::Stream { vad: false },
        language: None,
    },
    Case {
        name: "a13-whole-auto",
        sample: Sample::A13Excerpt,
        mode: Mode::Whole,
        language: None,
    },
];

fn samples_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples")
}

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

async fn load(sample: Sample) -> TestResult<Vec<f32>> {
    Ok(match sample {
        Sample::Jfk => {
            AudioProcessor::load_audio(samples_dir().join("jfk.wav"))
                .await?
                .samples
        }
        Sample::A13Excerpt => {
            let mut samples = AudioProcessor::load_audio(samples_dir().join("a13.mp3"))
                .await?
                .samples;
            samples.truncate(EXCERPT_SECONDS * SAMPLE_RATE);
            samples
        }
    })
}

fn config(vad: Option<PathBuf>, language: Option<&str>) -> TestResult<RunConfig> {
    Ok(RunConfig::new()
        .with_devices(DeviceOrder::cpu_only())
        .with_threads(THREADS)
        .with_opt_language(language.map(str::parse::<LanguageTag>).transpose()?)
        .with_vad_model(vad)
        .with_tentative(TentativeText::Decoded)
        .with_word_timestamps(true))
}

fn engines() -> Vec<Arc<dyn Engine>> {
    vec![Arc::new(WhisperEngine)]
}

fn event_json(event: &StreamEvent) -> TestResult<Value> {
    Ok(match event {
        StreamEvent::Chunk(chunk) => json!({
            "chunk": {
                "kind": serde_json::to_value(chunk.kind)?,
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
                "is_final": chunk.is_final,
                "chunk_index": chunk.chunk_index,
                "language": chunk.language.as_ref().map(LanguageTag::as_str),
                "words": serde_json::to_value(&chunk.words)?,
                "final_stats": chunk.final_stats.as_ref().map(|stats| json!({
                    "audio_duration": stats.audio_duration,
                    "segment_count": stats.segment_count,
                    "word_count": stats.word_count,
                })),
            }
        }),
        StreamEvent::Progress(progress) => json!({
            "progress": {
                "frame": serde_json::to_value(progress.frame)?,
                "received": progress.received,
                "source_ended": progress.source_ended,
            }
        }),
    })
}

async fn stream_run(model: &Path, config: RunConfig, samples: &[f32]) -> TestResult<Value> {
    let mut stream = transcribe(
        Input::Audio(AudioStream::from_samples(samples)),
        config,
        &ModelRef::new(model, EngineId::WhisperCpp),
        &engines(),
    )
    .await?;
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event_json(&event?)?);
    }
    Ok(Value::Array(events))
}

async fn whole_run(model: &Path, config: RunConfig, samples: &[f32]) -> TestResult<Value> {
    let result = collect(
        transcribe(
            Input::Audio(AudioStream::from_samples(samples)),
            config.with_windowing(Windowing::Whole),
            &ModelRef::new(model, EngineId::WhisperCpp),
            &engines(),
        )
        .await?,
    )
    .await?;
    Ok(json!({
        "text": result.text,
        "segments": result.segments.iter().map(|segment| Ok(json!({
            "text": segment.text,
            "start": segment.start,
            "end": segment.end,
            "words": serde_json::to_value(&segment.words)?,
        }))).collect::<TestResult<Vec<Value>>>()?,
        "audio_duration": result.audio_duration,
        "segment_count": result.stats.segment_count,
        "word_count": result.stats.word_count,
        "accel": serde_json::to_value(result.accel.selected)?,
    }))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs ggml-base.bin and the Silero model, and decodes about two minutes of audio on the CPU"]
async fn whisper_output_matches_the_goldens() -> TestResult {
    let models = ModelManager::new()?;
    let model = models.models_dir().join("ggml-base.bin");
    if !model.exists() {
        println!("skipping: no model at {}", model.display());
        return Ok(());
    }
    let vad = models.vad_model_path().await?;
    let Some(vad) = vad else {
        println!(
            "skipping: no Silero voice activity model in {}",
            models.models_dir().display()
        );
        return Ok(());
    };
    let update = std::env::var("PURR_GOLDEN").is_ok_and(|mode| mode == "update");

    let mut mismatched = Vec::new();
    for case in CASES {
        let samples = load(case.sample).await?;
        let recorded = match case.mode {
            Mode::Stream { vad: with_vad } => {
                let vad = with_vad.then(|| vad.clone());
                stream_run(&model, config(vad, case.language)?, &samples).await?
            }
            Mode::Whole => whole_run(&model, config(None, case.language)?, &samples).await?,
        };
        let path = golden_dir().join(format!("{}.json", case.name));
        if update {
            std::fs::create_dir_all(golden_dir())?;
            std::fs::write(&path, serde_json::to_string_pretty(&recorded)? + "\n")?;
            continue;
        }
        let golden: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        if golden != recorded {
            println!(
                "{} differs:\n{}",
                case.name,
                pretty_assertions::StrComparison::new(
                    &serde_json::to_string_pretty(&golden)?,
                    &serde_json::to_string_pretty(&recorded)?
                )
            );
            mismatched.push(case.name);
        }
    }
    assert_eq!(mismatched, Vec::<&str>::new());
    Ok(())
}
