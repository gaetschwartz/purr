#![allow(dead_code, reason = "every test binary uses a part of this module")]

use futures::StreamExt;
use purr_common::{devices::DeviceOrder, model::engine::EngineId, platform::ChunkKind};
use purr_core::{
    AudioStream, ModelManager,
    audio::AudioProcessor,
    config::{ModelRef, RunConfig, TentativeText, Windowing},
    engine::Engine,
    pipeline::{
        events::StreamEvent,
        result::{SyncTranscriptionResult, collect},
        run::{Input, transcribe},
    },
};
use purr_engine_transcribe::engine::TranscribeEngine;
use purr_golden::Recording;
use std::{
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};

pub type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

const SAMPLE_RATE: usize = 16_000;
const EXCERPT_SECONDS: usize = 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscribeModel {
    Parakeet,
    Qwen3,
}

impl TranscribeModel {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Parakeet => "parakeet-tdt-0.6b-v3-q4_k_m",
            Self::Qwen3 => "qwen3-asr-0.6b-q8_0",
        }
    }

    const fn slug(self) -> &'static str {
        match self {
            Self::Parakeet => "parakeet",
            Self::Qwen3 => "qwen3",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Sample {
    Jfk,
    Mm1,
    /// The first minute of a13.mp3
    A13Excerpt,
}

impl Sample {
    const fn slug(self) -> &'static str {
        match self {
            Self::Jfk => "jfk",
            Self::Mm1 => "mm1",
            Self::A13Excerpt => "a13",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Mode {
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

pub fn run_config() -> RunConfig {
    RunConfig::new()
        .with_devices(DeviceOrder::cpu_only())
        .with_threads(1)
}

pub fn engines() -> Vec<Arc<dyn Engine>> {
    vec![Arc::new(TranscribeEngine)]
}

pub enum MissingModel {
    Transcribe { id: &'static str, dir: PathBuf },
    VoiceActivity { dir: PathBuf },
}

impl fmt::Display for MissingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transcribe { id, dir } => write!(
                f,
                "the `{id}` model is missing or incomplete in {}: run `purr models download {id}` (built with `--features transcribe`) or `just download-test-models`",
                dir.display()
            ),
            Self::VoiceActivity { dir } => write!(
                f,
                "the Silero voice activity model is missing or incomplete in {}: run `purr models download base` or `just download-test-models`",
                dir.display()
            ),
        }
    }
}

impl fmt::Debug for MissingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for MissingModel {}

pub async fn model_path(model: TranscribeModel) -> TestResult<PathBuf> {
    let models = ModelManager::new()?;
    let spec = models.model(model.id())?;
    if !models.is_model_downloaded(spec).await? {
        return Err(MissingModel::Transcribe {
            id: model.id(),
            dir: models.models_dir().to_path_buf(),
        }
        .into());
    }
    Ok(models.get_model_path(spec))
}

pub async fn vad_model() -> TestResult<PathBuf> {
    let models = ModelManager::new()?;
    let path = models.vad_model_path().await?;
    path.ok_or_else(|| {
        MissingModel::VoiceActivity {
            dir: models.models_dir().to_path_buf(),
        }
        .into()
    })
}

pub async fn load(sample: Sample) -> TestResult<Vec<f32>> {
    let file = match sample {
        Sample::Jfk => "jfk.wav",
        Sample::Mm1 => "mm1.wav",
        Sample::A13Excerpt => "a13.mp3",
    };
    let mut samples = AudioProcessor::load_audio(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../samples")
            .join(file),
    )
    .await?
    .samples;
    if matches!(sample, Sample::A13Excerpt) {
        samples.truncate(EXCERPT_SECONDS * SAMPLE_RATE);
    }
    Ok(samples)
}

async fn stream_run(model: &Path, config: RunConfig, samples: &[f32]) -> TestResult<Recording> {
    let mut stream = transcribe(
        Input::Audio(AudioStream::from_samples(samples)),
        config,
        &ModelRef::new(model, EngineId::TranscribeCpp),
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

pub async fn whole_result(
    model: &Path,
    config: RunConfig,
    samples: &[f32],
) -> TestResult<SyncTranscriptionResult> {
    Ok(collect(
        transcribe(
            Input::Audio(AudioStream::from_samples(samples)),
            config.with_windowing(Windowing::Whole),
            &ModelRef::new(model, EngineId::TranscribeCpp),
            &engines(),
        )
        .await?,
    )
    .await?)
}

async fn whole_run(model: &Path, config: RunConfig, samples: &[f32]) -> TestResult<Recording> {
    let result = whole_result(model, config, samples).await?;
    Ok(Recording {
        text: result.text,
        audio_duration: Some(result.audio_duration),
    })
}

fn golden_path(model: TranscribeModel, sample: Sample, mode: Mode) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!(
            "{}-{}-{}.json",
            model.slug(),
            sample.slug(),
            mode.slug()
        ))
}

/// The text of the golden recorded for the whole-source run of `sample`.
pub fn golden_whole_text(model: TranscribeModel, sample: Sample) -> TestResult<String> {
    Ok(Recording::read(&golden_path(model, sample, Mode::Whole))?.text)
}

/// Runs the pipeline over `sample` and compares the words it produces with the recorded golden,
/// allowing `max_word_error_rate`;
/// `PURR_GOLDEN=update` records it again.
pub async fn assert_golden(
    model: TranscribeModel,
    sample: Sample,
    mode: Mode,
    max_word_error_rate: f64,
) -> TestResult {
    let model_path = model_path(model).await?;
    let samples = load(sample).await?;
    let config = run_config()
        .with_tentative(TentativeText::Decoded)
        .with_word_timestamps(true);
    let recorded = match mode {
        Mode::Stream => {
            let vad = vad_model().await?;
            stream_run(&model_path, config.with_vad_model(Some(vad)), &samples).await?
        }
        Mode::StreamWithoutVad => stream_run(&model_path, config, &samples).await?,
        Mode::Whole => whole_run(&model_path, config, &samples).await?,
    };
    purr_golden::check(
        &golden_path(model, sample, mode),
        &recorded,
        max_word_error_rate,
    )?;
    Ok(())
}
