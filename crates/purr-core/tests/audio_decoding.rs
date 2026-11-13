//! Decoded length of compressed audio, through the whole-file and the streaming loaders.
//!
//! Needs FFmpeg only: no model, no GPU.

use ffmpeg_next as ffmpeg;
use futures::StreamExt;
use purr_core::audio::AudioProcessor;
use std::path::{Path, PathBuf};

const SAMPLE_RATE: i64 = 16_000;

/// Codec priming and end padding (MP3 encoder delay is about 1100 samples at 44.1 kHz, Opus
/// pre-skip is 312 samples at 48 kHz) stay well below 100 ms.
const TOLERANCE_SAMPLES: i64 = SAMPLE_RATE / 10;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn sample_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("samples")
        .join(name)
}

fn container_samples(path: &Path) -> Result<i64, Box<dyn std::error::Error>> {
    ffmpeg::init()?;
    let input = ffmpeg::format::input(&path)?;
    let micros = input.duration();
    let ticks_per_second = i64::from(ffmpeg::sys::AV_TIME_BASE);
    Ok(micros * SAMPLE_RATE / ticks_per_second)
}

async fn streamed_samples(path: &Path) -> Result<i64, Box<dyn std::error::Error>> {
    let mut stream = AudioProcessor::stream_file(path).await?;
    let mut total = 0usize;
    while let Some(chunk) = stream.next().await {
        total += chunk?.samples.len();
    }
    Ok(i64::try_from(total)?)
}

async fn assert_decoded_length(name: &str) -> TestResult {
    let path = sample_path(name);
    let expected = container_samples(&path)?;

    let loaded = AudioProcessor::load_audio(&path).await?;
    let loaded_samples = i64::try_from(loaded.samples.len())?;
    let streamed = streamed_samples(&path).await?;

    assert!(
        (loaded_samples - expected).abs() <= TOLERANCE_SAMPLES,
        "{name}: load_audio decoded {loaded_samples} samples, container says {expected}"
    );
    assert!(
        (streamed - expected).abs() <= TOLERANCE_SAMPLES,
        "{name}: stream_file decoded {streamed} samples, container says {expected}"
    );
    assert!(
        (loaded_samples - streamed).abs() <= TOLERANCE_SAMPLES,
        "{name}: load_audio decoded {loaded_samples} samples, stream_file {streamed}"
    );
    Ok(())
}

#[tokio::test]
async fn decodes_the_whole_mp3() -> TestResult {
    assert_decoded_length("jfk.mp3").await
}

#[tokio::test]
async fn decodes_the_whole_ogg() -> TestResult {
    assert_decoded_length("jfk.ogg").await
}

#[tokio::test]
async fn decodes_the_whole_opus() -> TestResult {
    assert_decoded_length("jfk.opus").await
}

#[tokio::test]
async fn decodes_the_whole_wav() -> TestResult {
    assert_decoded_length("jfk.wav").await
}

#[tokio::test]
async fn decodes_the_whole_long_ogg() -> TestResult {
    assert_decoded_length("gb1.ogg").await
}
