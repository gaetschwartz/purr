//! Whole-file transcription of known samples and broken inputs.

use purr_common::{language::tag::LanguageTag, model::engine::EngineId};
use purr_core::{
    Error, ModelManager,
    config::{ModelRef, RunConfig, Windowing},
    engine::Engine,
    pipeline::{
        result::{SyncTranscriptionResult, collect},
        run::{Input, transcribe},
    },
};
use purr_engine_whisper::engine::WhisperEngine;
use rstest::rstest;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

async fn default_model() -> Option<PathBuf> {
    ModelManager::new()
        .ok()?
        .find_default_model(&[purr_common::model::engine::EngineId::WhisperCpp])
        .await
        .ok()
}

async fn transcribe_file_sync(
    path: &str,
    config: RunConfig,
    model: &Path,
) -> purr_core::Result<SyncTranscriptionResult> {
    let engines: Vec<Arc<dyn Engine>> = vec![Arc::new(WhisperEngine)];
    collect(
        transcribe(
            Input::File(PathBuf::from(path)),
            config.with_windowing(Windowing::Whole),
            &ModelRef::new(model, EngineId::WhisperCpp),
            &engines,
        )
        .await?,
    )
    .await
}

#[test]
fn test_transcription_config() {
    let config = RunConfig::new()
        .with_language(LanguageTag::english())
        .with_gpu(false)
        .with_threads(2);

    assert_eq!(config.language, Some(LanguageTag::english()));
    assert_eq!(
        config.devices,
        purr_common::devices::DeviceOrder::cpu_only()
    );
    assert_eq!(config.threads, Some(2));
}

#[rstest]
#[case(RunConfig::new().with_gpu(false).with_language(LanguageTag::english()))]
#[case(RunConfig::new().with_gpu(false).with_threads(1))]
#[case(RunConfig::new().with_gpu(false).with_word_timestamps(true))]
#[tokio::test]
async fn test_transcription_configurations(#[case] config: RunConfig) {
    let Some(model) = default_model().await else {
        println!("⏭ Skipping - no model available");
        return;
    };
    let transcription = transcribe_file_sync("../../samples/jfk.wav", config, &model)
        .await
        .unwrap();
    assert_ne!(transcription.text, "");
    assert!(transcription.audio_duration > 0.0);
    assert!(!transcription.segments.is_empty());
    for segment in &transcription.segments {
        assert!(segment.start >= 0.0);
        assert!(segment.end >= segment.start);
    }
}

#[rstest]
#[case("nonexistent.wav")]
#[case("../../Cargo.toml")]
#[tokio::test]
async fn test_transcription_error_handling(#[case] invalid_path: &str) {
    let Some(model) = default_model().await else {
        println!("⏭ Skipping - no model available");
        return;
    };
    let result = transcribe_file_sync(invalid_path, RunConfig::new().with_gpu(false), &model).await;
    assert!(
        matches!(
            result,
            Err(Error::AudioProcessing { .. } | Error::Io { .. } | Error::FFmpeg { .. })
        ),
        "{result:?}"
    );
}
