//! Whole-file transcription of every sample.

use purr_common::model::engine::EngineId;
use purr_core::{
    ModelManager,
    config::{ModelRef, RunConfig, Windowing},
    engine::Engine,
    pipeline::{
        result::collect,
        run::{Input, transcribe},
    },
};
use purr_engine_whisper::engine::WhisperEngine;
use rstest::rstest;
use std::{path::PathBuf, sync::Arc};

#[rstest]
#[tokio::test]
async fn test_transcribe_sample_files(#[files("../../samples/*")] sample_path: PathBuf) {
    let extension = sample_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();
    if !matches!(extension.as_str(), "wav" | "mp3" | "ogg" | "flac") {
        println!("â­ Skipping {}", sample_path.display());
        return;
    }
    let Some(model) = (match ModelManager::new() {
        Ok(manager) => manager
            .find_default_model(&[purr_common::model::engine::EngineId::WhisperCpp])
            .await
            .ok(),
        Err(_) => None,
    }) else {
        println!("â­ Skipping {} - no model available", sample_path.display());
        return;
    };

    let engines: Vec<Arc<dyn Engine>> = vec![Arc::new(WhisperEngine)];
    let transcription = collect(
        transcribe(
            Input::File(sample_path.clone()),
            RunConfig::new()
                .with_gpu(false)
                .with_windowing(Windowing::Whole),
            &ModelRef::new(model, EngineId::WhisperCpp),
            &engines,
        )
        .await
        .unwrap(),
    )
    .await
    .unwrap();

    assert_ne!(transcription.text, "", "{}", sample_path.display());
    assert!(transcription.audio_duration > 0.0);
    assert!(transcription.processing_time > 0.0);
    assert!(!transcription.segments.is_empty());
    for segment in &transcription.segments {
        assert!(!segment.text.is_empty());
        assert!(segment.start >= 0.0);
        assert!(segment.end >= segment.start);
    }
    assert!(
        transcription
            .segments
            .windows(2)
            .all(|pair| pair[0].start <= pair[1].start)
    );
}
