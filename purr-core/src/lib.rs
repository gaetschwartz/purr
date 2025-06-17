//! Whisper UI Core Library
//!
//! This library provides audio transcription functionality using whisper.cpp and FFmpeg.

pub mod audio;
pub mod config;
pub mod dev;
pub mod error;
pub mod math;
pub mod model;
pub mod whisper;

pub use audio::{AudioChunk, AudioProcessor, AudioStream};
pub use config::TranscriptionConfig;
pub use dev::{list_devices, Device, SystemInfo};
pub use error::{Result, WhisperError};
pub use model::{ModelManager, WhisperModel};
use tokio::try_join;
use tracing::info;
pub use whisper::logging::install_logging_hooks;

use crate::whisper::{
    streaming::StreamWhisperTranscriber, sync::SyncWhisperTranscriber, WhisperTranscriber,
};

// Re-export public types from whisper module for CLI
pub use whisper::streaming::StreamingTranscriptionResult;
pub use whisper::{StreamingChunk, SyncTranscriptionResult};

/// High-level transcription function
pub async fn transcribe_file_sync<P: AsRef<std::path::Path>>(
    audio_path: P,
    config: Option<TranscriptionConfig>,
) -> Result<SyncTranscriptionResult> {
    let config = config.unwrap_or_default();

    // Initialize transcriber
    let transcriber = SyncWhisperTranscriber::from_config(config).await?;

    info!("Transcribing audio file: {:?}", audio_path.as_ref());
    // Process audio
    let mut audio_processor = AudioProcessor::new()?;
    let audio_data = audio_processor.load_audio(audio_path).await?;

    info!("Audio data loaded, starting transcription...");

    // Transcribe
    transcriber.transcribe(audio_data).await
}

/// True streaming transcription function that processes audio in chunks
pub async fn transcribe_file_stream<P: AsRef<std::path::Path>>(
    audio_path: P,
    config: Option<TranscriptionConfig>,
) -> Result<StreamingTranscriptionResult> {
    let config = config.unwrap_or_default();

    info!(
        "Starting real-time streaming transcription for: {:?}",
        audio_path.as_ref()
    );

    // Initialize transcriber
    let (transcriber, audio_stream) = try_join!(
        StreamWhisperTranscriber::from_config(config),
        AudioProcessor::stream(audio_path)
    )?;

    info!("Audio stream created, starting transcription...");

    // Start streaming transcription (consumes both transcriber and stream)
    transcriber.transcribe(audio_stream).await
}
