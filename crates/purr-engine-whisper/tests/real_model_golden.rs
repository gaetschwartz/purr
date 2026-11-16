//! What the Whisper pipeline produces on samples of different lengths, on `base` and `small`,
//! compared with recorded goldens.
//!
//! `PURR_GOLDEN=update` records the goldens again.

mod common;

use common::WhisperModel;
use futures::StreamExt;
use purr_common::{language::tag::LanguageTag, model::engine::EngineId};
use purr_core::{
    AudioStream,
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
use rstest::rstest;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

const SAMPLE_RATE: usize = 16_000;
const EXCERPT_SECONDS: usize = 60;

#[derive(Debug, Clone, Copy)]
enum Sample {
    Jfk,
    Mm1,
    /// The first minute of a13.mp3
    A13Excerpt,
    A13,
}

impl Sample {
    const fn slug(self) -> &'static str {
        match self {
            Self::Jfk => "jfk",
            Self::Mm1 => "mm1",
            Self::A13Excerpt => "a13",
            Self::A13 => "a13-full",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Mode {
    Stream,
    StreamWithoutVad,
    Whole,
}

impl Mode {
    const fn slug(self) -> &'static str {
        match self {
            Self::Stream => "stream",
            Self::StreamWithoutVad => "stream-novad",
            Self::Whole => "whole",
        }
    }
}

fn samples_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples")
}

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

async fn load(sample: Sample) -> TestResult<Vec<f32>> {
    let file = match sample {
        Sample::Jfk => "jfk.wav",
        Sample::Mm1 => "mm1.wav",
        Sample::A13Excerpt | Sample::A13 => "a13.mp3",
    };
    let mut samples = AudioProcessor::load_audio(samples_dir().join(file))
        .await?
        .samples;
    if matches!(sample, Sample::A13Excerpt) {
        samples.truncate(EXCERPT_SECONDS * SAMPLE_RATE);
    }
    Ok(samples)
}

fn config(vad: Option<PathBuf>, language: Option<&str>) -> TestResult<RunConfig> {
    Ok(common::run_config()
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

#[rstest]
#[case::base_jfk_stream_auto(WhisperModel::Base, Sample::Jfk, Mode::Stream, None)]
#[case::base_jfk_stream_en(WhisperModel::Base, Sample::Jfk, Mode::Stream, Some("en"))]
#[case::base_jfk_whole_auto(WhisperModel::Base, Sample::Jfk, Mode::Whole, None)]
#[case::base_jfk_whole_en(WhisperModel::Base, Sample::Jfk, Mode::Whole, Some("en"))]
#[case::base_mm1_stream_auto(WhisperModel::Base, Sample::Mm1, Mode::Stream, None)]
#[case::base_mm1_whole_auto(WhisperModel::Base, Sample::Mm1, Mode::Whole, None)]
#[case::base_a13_stream_auto(WhisperModel::Base, Sample::A13Excerpt, Mode::Stream, None)]
#[case::base_a13_stream_novad_auto(
    WhisperModel::Base,
    Sample::A13Excerpt,
    Mode::StreamWithoutVad,
    None
)]
#[case::base_a13_whole_auto(WhisperModel::Base, Sample::A13Excerpt, Mode::Whole, None)]
#[case::base_a13_full_stream_auto(WhisperModel::Base, Sample::A13, Mode::Stream, None)]
#[case::small_jfk_stream_auto(WhisperModel::Small, Sample::Jfk, Mode::Stream, None)]
#[case::small_jfk_stream_en(WhisperModel::Small, Sample::Jfk, Mode::Stream, Some("en"))]
#[case::small_jfk_whole_auto(WhisperModel::Small, Sample::Jfk, Mode::Whole, None)]
#[case::small_jfk_whole_en(WhisperModel::Small, Sample::Jfk, Mode::Whole, Some("en"))]
#[case::small_mm1_stream_auto(WhisperModel::Small, Sample::Mm1, Mode::Stream, None)]
#[case::small_mm1_whole_auto(WhisperModel::Small, Sample::Mm1, Mode::Whole, None)]
#[case::small_a13_stream_auto(WhisperModel::Small, Sample::A13Excerpt, Mode::Stream, None)]
#[case::small_a13_stream_novad_auto(
    WhisperModel::Small,
    Sample::A13Excerpt,
    Mode::StreamWithoutVad,
    None
)]
#[case::small_a13_whole_auto(WhisperModel::Small, Sample::A13Excerpt, Mode::Whole, None)]
#[case::small_a13_full_stream_auto(WhisperModel::Small, Sample::A13, Mode::Stream, None)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn whisper_output_matches_the_golden(
    #[case] model: WhisperModel,
    #[case] sample: Sample,
    #[case] mode: Mode,
    #[case] language: Option<&str>,
) -> TestResult {
    let model_path = common::model_path(model).await?;
    let samples = load(sample).await?;
    let recorded = match mode {
        Mode::Stream => {
            let vad = common::vad_model().await?;
            stream_run(&model_path, config(Some(vad), language)?, &samples).await?
        }
        Mode::StreamWithoutVad => {
            stream_run(&model_path, config(None, language)?, &samples).await?
        }
        Mode::Whole => whole_run(&model_path, config(None, language)?, &samples).await?,
    };
    let name = format!(
        "{}-{}-{}-{}",
        model.id(),
        sample.slug(),
        mode.slug(),
        language.unwrap_or("auto")
    );
    let path = golden_dir().join(format!("{name}.json"));
    if std::env::var("PURR_GOLDEN").is_ok_and(|mode| mode == "update") {
        std::fs::write(&path, serde_json::to_string_pretty(&recorded)? + "\n")?;
        return Ok(());
    }
    let golden =
        std::fs::read_to_string(&path).map_err(|source| common::MissingGolden { path, source })?;
    pretty_assertions::assert_eq!(
        serde_json::to_string_pretty(&serde_json::from_str::<Value>(&golden)?)?,
        serde_json::to_string_pretty(&recorded)?
    );
    Ok(())
}
