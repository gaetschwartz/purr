//! Properties of the Whisper pipeline that the goldens do not pin: a repeated sentence is kept,
//! and progress follows the audio consumed.

mod common;

use common::WhisperModel;
use futures::StreamExt;
use purr_common::{
    devices::DeviceOrder,
    language::tag::LanguageTag,
    model::{engine::EngineId, profile::Timing},
    progress::ProgressPhase,
};
use purr_core::{
    AudioStream,
    audio::AudioProcessor,
    config::{ModelRef, RunConfig},
    engine::{
        Engine,
        cancel::CancelToken,
        decoder::{DecodeRequest, LanguageRequest},
        open::open_on_best_device,
    },
    pipeline::{
        events::{StreamEvent, StreamingChunk, Transcription},
        run::{Input, transcribe},
    },
};
use purr_engine_whisper::engine::WhisperEngine;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

const JFK_SENTENCE: &str = "ask what you can do for your country";

fn engines() -> Vec<Arc<dyn Engine>> {
    vec![Arc::new(WhisperEngine)]
}

/// The sample spoken twice, two seconds apart.
async fn spoken_twice(sample: &str) -> TestResult<AudioStream> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../samples")
        .join(sample);
    let audio = AudioProcessor::load_audio(&path).await?;
    let silence = vec![0.0; audio.sample_rate as usize * 2];
    let samples = [audio.samples.as_slice(), &silence, &audio.samples].concat();
    Ok(AudioStream::from_samples(&samples))
}

async fn stream(
    audio: AudioStream,
    config: RunConfig,
    model: &Path,
) -> purr_core::Result<Transcription> {
    transcribe(
        Input::Audio(audio),
        config,
        &ModelRef::new(model, EngineId::WhisperCpp),
        &engines(),
    )
    .await
}

async fn chunks(
    audio: AudioStream,
    config: RunConfig,
    model: &Path,
) -> TestResult<Vec<StreamingChunk>> {
    let mut events = stream(audio, config, model).await?;
    let mut chunks = Vec::new();
    while let Some(event) = events.next().await {
        if let StreamEvent::Chunk(chunk) = event? {
            chunks.push(chunk);
        }
    }
    Ok(chunks)
}

/// How often the run of words `phrase` occurs in `text`, ignoring case and punctuation.
fn count_run(text: &str, phrase: &str) -> usize {
    let words = |text: &str| -> Vec<String> {
        text.split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|c| c.is_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect::<String>()
            })
            .filter(|word| !word.is_empty())
            .collect()
    };
    let (text, phrase) = (words(text), words(phrase));
    text.windows(phrase.len())
        .filter(|run| *run == phrase.as_slice())
        .count()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sentence_spoken_twice_is_kept_twice_with_and_without_voice_activity() -> TestResult {
    let model = common::model_path(WhisperModel::Base).await?;
    let vad = common::vad_model().await?;
    let config = common::run_config();

    for config in [config.clone(), config.with_vad_model(Some(vad))] {
        let chunks = chunks(spoken_twice("jfk.wav").await?, config, &model).await?;
        assert!(chunks.last().is_some_and(|chunk| chunk.is_final));
        assert!(
            chunks
                .windows(2)
                .all(|pair| pair[0].end <= pair[1].start + 0.01)
        );
        let stats = chunks
            .last()
            .and_then(|chunk| chunk.final_stats.clone())
            .expect("the last chunk carries the statistics");
        assert_eq!(
            stats.segment_count,
            chunks.iter().filter(|c| !c.text.is_empty()).count()
        );
        let text = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(count_run(&text, JFK_SENTENCE), 2, "{text}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn progress_follows_the_audio_consumed() -> TestResult {
    let model = common::model_path(WhisperModel::Base).await?;
    let mut events = stream(spoken_twice("jfk.wav").await?, common::run_config(), &model).await?;
    let mut frames = Vec::new();
    while let Some(event) = events.next().await {
        if let StreamEvent::Progress(progress) = event? {
            frames.push(progress);
        }
    }

    let done: Vec<f64> = frames.iter().map(|p| p.frame.transcribed_to).collect();
    assert!(done.windows(2).all(|pair| pair[0] <= pair[1]), "{done:?}");
    assert!(frames.iter().all(|p| p.frame.transcribed_to <= p.received));
    for (progress, (start, end)) in frames
        .iter()
        .filter_map(|p| p.frame.decoding.map(|d| (p, d)))
    {
        assert_eq!(start, progress.frame.transcribed_to);
        assert!(end > start && end <= progress.received);
    }
    let last = frames.last().expect("progress is reported");
    assert!(last.source_ended);
    assert_eq!(last.frame.decoding, None);
    assert_eq!(last.frame.phase, ProgressPhase::Transcribing);
    assert_eq!(last.frame.transcribed_to, last.received);
    assert!((last.received - 24.0).abs() < 0.1, "{}", last.received);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_window_decodes_the_same_after_other_decodes() -> TestResult {
    let model = common::model_path(WhisperModel::Base).await?;
    let audio = AudioProcessor::load_audio(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../samples/jfk.wav"),
    )
    .await?;
    let mut decoder =
        open_on_best_device(&WhisperEngine, &model, &DeviceOrder::cpu_only(), 1)?.decoder;
    let language = LanguageRequest::Use(LanguageTag::english());
    let cancel = CancelToken::new();
    let mut decode = |pcm: &[f32]| {
        decoder.decode(DecodeRequest {
            pcm,
            language: &language,
            timing: Timing::Segment,
            cancel: &cancel,
        })
    };

    let first = decode(&audio.samples)?;
    for len in [32_000, 88_000, 160_000, audio.samples.len()] {
        decode(&audio.samples[..len])?;
        assert_eq!(decode(&audio.samples)?, first);
    }
    Ok(())
}
