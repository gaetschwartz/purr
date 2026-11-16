//! What the Whisper pipeline produces on samples of different lengths, on `base` and `small`,
//! compared with recorded goldens by their words.
//!
//! `PURR_GOLDEN=update` records the goldens again.

mod common;

use common::WhisperModel;
use futures::StreamExt;
use purr_common::{language::tag::LanguageTag, model::engine::EngineId, platform::ChunkKind};
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
use purr_golden::Recording;
use rstest::rstest;
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

async fn stream_run(model: &Path, config: RunConfig, samples: &[f32]) -> TestResult<Recording> {
    let mut stream = transcribe(
        Input::Audio(AudioStream::from_samples(samples)),
        config,
        &ModelRef::new(model, EngineId::WhisperCpp),
        &engines(),
    )
    .await?;
    let mut committed = Vec::new();
    while let Some(event) = stream.next().await {
        if let StreamEvent::Chunk(chunk) = event?
            && chunk.kind == ChunkKind::Committed
        {
            committed.push(chunk.text);
        }
    }
    Ok(Recording {
        text: committed.join(" "),
        audio_duration: None,
    })
}

async fn whole_run(model: &Path, config: RunConfig, samples: &[f32]) -> TestResult<Recording> {
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
    Ok(Recording {
        text: result.text,
        audio_duration: Some(result.audio_duration),
    })
}

#[rstest]
#[case::base_jfk_stream_auto(WhisperModel::Base, Sample::Jfk, Mode::Stream, None, 0.0)]
#[case::base_jfk_stream_en(WhisperModel::Base, Sample::Jfk, Mode::Stream, Some("en"), 0.0)]
#[case::base_jfk_whole_auto(WhisperModel::Base, Sample::Jfk, Mode::Whole, None, 0.0)]
#[case::base_jfk_whole_en(WhisperModel::Base, Sample::Jfk, Mode::Whole, Some("en"), 0.0)]
#[case::base_mm1_stream_auto(WhisperModel::Base, Sample::Mm1, Mode::Stream, None, 0.14)]
#[case::base_mm1_whole_auto(WhisperModel::Base, Sample::Mm1, Mode::Whole, None, 0.14)]
#[case::base_a13_stream_auto(WhisperModel::Base, Sample::A13Excerpt, Mode::Stream, None, 0.11)]
#[case::base_a13_stream_novad_auto(
    WhisperModel::Base,
    Sample::A13Excerpt,
    Mode::StreamWithoutVad,
    None,
    0.05
)]
#[case::base_a13_whole_auto(WhisperModel::Base, Sample::A13Excerpt, Mode::Whole, None, 0.05)]
#[case::base_a13_full_stream_auto(WhisperModel::Base, Sample::A13, Mode::Stream, None, 0.08)]
#[case::small_jfk_stream_auto(WhisperModel::Small, Sample::Jfk, Mode::Stream, None, 0.0)]
#[case::small_jfk_stream_en(WhisperModel::Small, Sample::Jfk, Mode::Stream, Some("en"), 0.0)]
#[case::small_jfk_whole_auto(WhisperModel::Small, Sample::Jfk, Mode::Whole, None, 0.0)]
#[case::small_jfk_whole_en(WhisperModel::Small, Sample::Jfk, Mode::Whole, Some("en"), 0.0)]
#[case::small_mm1_stream_auto(WhisperModel::Small, Sample::Mm1, Mode::Stream, None, 0.07)]
#[case::small_mm1_whole_auto(WhisperModel::Small, Sample::Mm1, Mode::Whole, None, 0.07)]
#[case::small_a13_stream_auto(WhisperModel::Small, Sample::A13Excerpt, Mode::Stream, None, 0.0)]
#[case::small_a13_stream_novad_auto(
    WhisperModel::Small,
    Sample::A13Excerpt,
    Mode::StreamWithoutVad,
    None,
    0.0
)]
#[case::small_a13_whole_auto(WhisperModel::Small, Sample::A13Excerpt, Mode::Whole, None, 0.0)]
#[case::small_a13_full_stream_auto(WhisperModel::Small, Sample::A13, Mode::Stream, None, 0.11)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn whisper_output_matches_the_golden(
    #[case] model: WhisperModel,
    #[case] sample: Sample,
    #[case] mode: Mode,
    #[case] language: Option<&str>,
    #[case] max_word_error_rate: f64,
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
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.json"));
    purr_golden::check(&path, &recorded, max_word_error_rate)?;
    Ok(())
}
