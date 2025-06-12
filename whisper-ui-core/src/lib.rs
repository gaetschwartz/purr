//! Whisper UI Core Library
//!
//! This library provides audio transcription functionality using whisper.cpp and FFmpeg.

pub mod audio;
pub mod config;
pub mod error;
pub mod simple_audio;
pub mod transcription;

pub use audio::AudioProcessor;
pub use config::TranscriptionConfig;
pub use error::{Result, WhisperError};
pub use simple_audio::SimpleAudioProcessor;
pub use transcription::{TranscriptionResult, WhisperTranscriber};

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
