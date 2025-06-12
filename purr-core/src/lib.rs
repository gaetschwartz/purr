//! Whisper UI Core Library
//!
//! This library provides audio transcription functionality using whisper.cpp and FFmpeg.

pub mod audio;
pub mod config;
pub mod error;
pub mod gpu;
pub mod model;
pub mod simple_audio;
pub mod transcription;

pub use audio::AudioProcessor;
pub use config::TranscriptionConfig;
pub use error::{Result, WhisperError};
pub use gpu::{GpuDevice, GpuStatus, check_gpu_status, list_gpu_devices};
pub use model::{ModelManager, WhisperModel};
pub use simple_audio::SimpleAudioProcessor;
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

    // Process audio
    let mut audio_processor = AudioProcessor::new();
    let audio_data = audio_processor.load_audio(audio_path).await?;

    // Transcribe
    transcriber.transcribe(audio_data).await
}

/// High-level streaming transcription function
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
