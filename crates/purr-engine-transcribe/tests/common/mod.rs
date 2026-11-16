#![allow(dead_code, reason = "every test binary uses a part of this module")]

use futures::StreamExt;
use purr_common::{
    devices::DeviceOrder, language::tag::LanguageTag, model::engine::EngineId, platform::ChunkKind,
};
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
use serde_json::{Value, json};
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

pub struct MissingGolden {
    pub path: PathBuf,
    pub source: std::io::Error,
}

impl fmt::Display for MissingGolden {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot read the golden {}: record it with PURR_GOLDEN=update",
            self.path.display()
        )
    }
}

impl fmt::Debug for MissingGolden {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for MissingGolden {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

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

async fn stream_run(model: &Path, config: RunConfig, samples: &[f32]) -> TestResult<Value> {
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
            chunks.push(json!({
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
                "language": chunk.language.as_ref().map(LanguageTag::as_str),
                "words": serde_json::to_value(&chunk.words)?,
            }));
        }
    }
    Ok(Value::Array(chunks))
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

async fn whole_run(model: &Path, config: RunConfig, samples: &[f32]) -> TestResult<Value> {
    let result = whole_result(model, config, samples).await?;
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
    }))
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
    let path = golden_path(model, sample, Mode::Whole);
    let golden = std::fs::read_to_string(&path).map_err(|source| MissingGolden { path, source })?;
    let golden: Value = serde_json::from_str(&golden)?;
    let text = golden
        .get("text")
        .and_then(Value::as_str)
        .ok_or("the golden has no text")?;
    Ok(text.to_owned())
}

/// Runs the pipeline over `sample` and compares what it produces with the recorded golden;
/// `PURR_GOLDEN=update` records it again.
pub async fn assert_golden(model: TranscribeModel, sample: Sample, mode: Mode) -> TestResult {
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

    let path = golden_path(model, sample, mode);
    if std::env::var("PURR_GOLDEN").is_ok_and(|mode| mode == "update") {
        std::fs::write(&path, serde_json::to_string_pretty(&recorded)? + "\n")?;
        return Ok(());
    }
    let golden = std::fs::read_to_string(&path).map_err(|source| MissingGolden { path, source })?;
    pretty_assertions::assert_eq!(
        serde_json::to_string_pretty(&serde_json::from_str::<Value>(&golden)?)?,
        serde_json::to_string_pretty(&recorded)?
    );
    Ok(())
}
