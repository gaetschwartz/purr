//! Whisper UI Core Library
//!
//! This library provides audio transcription functionality using whisper.cpp and FFmpeg.

pub mod audio;
pub mod config;
pub mod dev;
pub mod error;
pub mod math;
pub mod model;
pub mod transcription;

pub use audio::{AudioChunk, AudioProcessor, AudioStream};
pub use config::TranscriptionConfig;
pub use dev::{check_gpu_status, list_devices, Device, GpuStatus};
pub use error::{Result, WhisperError};
pub use model::{ModelManager, WhisperModel};
use tracing::info;
pub use transcription::{
    StreamingChunk, StreamingReceiver, TranscriptionResult, WhisperTranscriber,
};


/// High-level transcription function
pub async fn transcribe_audio_file<P: AsRef<std::path::Path>>(
    audio_path: P,
    config: Option<TranscriptionConfig>,
) -> Result<TranscriptionResult> {
    let config = config.unwrap_or_default();

    // Initialize transcriber
    let mut transcriber = WhisperTranscriber::new(config.clone()).await?;

    info!("Transcribing audio file: {:?}", audio_path.as_ref());
    // Process audio
    let mut audio_processor = AudioProcessor::new();
    let audio_data = audio_processor.load_audio(audio_path).await?;

    info!("Audio data loaded, starting transcription...");

    // Transcribe
    transcriber.transcribe(audio_data).await
}

/// High-level streaming transcription function (simulated streaming)
pub async fn transcribe_audio_file_streaming<P: AsRef<std::path::Path>>(
    audio_path: P,
    config: Option<TranscriptionConfig>,
) -> Result<StreamingReceiver> {
    let config = config.unwrap_or_default();

    // Initialize transcriber
    let transcriber = WhisperTranscriber::new(config.clone()).await?;

    // Process audio
    let mut audio_processor = AudioProcessor::new();
    let audio_data = audio_processor.load_audio(audio_path).await?;

    // Start streaming transcription (consumes the transcriber)
    transcriber.transcribe_streaming(audio_data).await
}

/// True streaming transcription function that processes audio in chunks
pub async fn transcribe_audio_file_streaming_realtime<P: AsRef<std::path::Path>>(
    audio_path: P,
    config: Option<TranscriptionConfig>,
) -> Result<StreamingReceiver> {
    let config = config.unwrap_or_default();

    info!("Starting real-time streaming transcription for: {:?}", audio_path.as_ref());

    // Initialize transcriber
    let transcriber = WhisperTranscriber::new(config.clone()).await?;

    // Start streaming audio processing
    let mut audio_processor = AudioProcessor::new();
    let audio_stream = audio_processor.stream_audio(audio_path).await?;

    info!("Audio stream created, starting transcription...");

    // Start streaming transcription (consumes both transcriber and stream)
    transcriber.transcribe_audio_stream(audio_stream).await
}
