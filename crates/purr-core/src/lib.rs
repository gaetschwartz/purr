//! Whisper UI Core Library
//!
//! This library provides audio transcription functionality using whisper.cpp and `FFmpeg`.

pub mod audio;
pub mod config;
pub mod dev;
pub mod error;
pub mod input;
pub mod math;
pub mod model;
pub mod url;
pub mod whisper;

use crate::whisper::{
    streaming::StreamWhisperTranscriber, sync::SyncWhisperTranscriber, WhisperTranscriber,
};
pub use audio::{AudioChunk, AudioProcessor, AudioStream};
pub use config::TranscriptionConfig;
pub use dev::{list_devices, Device, SystemInfo};
pub use error::{Result, WhisperError};
pub use model::{ModelManager, WhisperModel};
use reqwest::IntoUrl;
use tokio::try_join;
use tracing::debug;
pub use url::{is_file_path, is_valid_url, HttpStreamer, UrlContentInfo, UrlStreamConfig};
pub use whisper::logging::install_logging_hooks;
pub use whisper::streaming::StreamingTranscription;
pub use whisper::{StreamingChunk, SyncTranscriptionResult};
/// High-level transcription function
pub async fn transcribe_file_sync<P: AsRef<std::path::Path>>(
    audio_path: P,
    config: Option<TranscriptionConfig>,
) -> Result<SyncTranscriptionResult> {
    let config = config.unwrap_or_default();

    // Initialize transcriber
    let transcriber = SyncWhisperTranscriber::from_config(config).await?;

    debug!("Transcribing audio file: {:?}", audio_path.as_ref());
    // Process audio
    let audio_data = AudioProcessor::load_audio(audio_path).await?;

    debug!("Audio data loaded, starting transcription...");

    // Transcribe
    transcriber.transcribe(audio_data).await
}

/// True streaming transcription function that processes audio in chunks
pub async fn transcribe_file_stream<P: AsRef<std::path::Path>>(
    audio_path: P,
    config: TranscriptionConfig,
) -> Result<StreamingTranscription> {
    debug!(
        "Starting real-time streaming transcription for: {:?}",
        audio_path.as_ref()
    );

    // Initialize transcriber
    let (transcriber, audio_stream) = try_join!(
        StreamWhisperTranscriber::from_config(config),
        AudioProcessor::stream_file(audio_path)
    )?;

    debug!("Audio stream created, starting transcription...");

    // Start streaming transcription (consumes both transcriber and stream)
    transcriber.transcribe(audio_stream).await
}

/// Synchronous transcription function for URLs
pub async fn transcribe_url_sync(
    url: impl IntoUrl,
    config: Option<TranscriptionConfig>,
) -> Result<SyncTranscriptionResult> {
    let config = config.unwrap_or_default();
    let url = url.into_url()?;

    debug!("Transcribing audio from URL: {}", url);

    // Initialize transcriber
    let transcriber = SyncWhisperTranscriber::from_config(config.clone()).await?;

    // Stream and process audio from URL
    let audio_data = AudioProcessor::load_audio_from_url_with_config(url, &config).await?;

    debug!("Audio data loaded from URL, starting transcription...");

    // Transcribe
    transcriber.transcribe(audio_data).await
}

/// Streaming transcription function for URLs
pub async fn transcribe_url_stream(
    url: impl IntoUrl,
    config: TranscriptionConfig,
) -> Result<StreamingTranscription> {
    let url = url.into_url()?;
    debug!(
        "Starting real-time streaming transcription for URL: {}",
        url
    );

    // Initialize transcriber
    let (transcriber, audio_stream) = try_join!(
        StreamWhisperTranscriber::from_config(config.clone()),
        AudioProcessor::stream_url_with_config(url, &config)
    )?;

    debug!("Audio stream created from URL, starting transcription...");

    // Start streaming transcription (consumes both transcriber and stream)
    transcriber.transcribe(audio_stream).await
}

pub const PKG_NAME: &str = env!("CARGO_PKG_NAME");
