//! Whisper UI Core Library
//! 
//! This library provides audio transcription functionality using whisper.cpp and FFmpeg.

pub mod audio;
pub mod transcription;
pub mod error;
pub mod config;

pub use audio::AudioProcessor;
pub use transcription::{WhisperTranscriber, TranscriptionResult};
pub use error::{WhisperError, Result};
pub use config::TranscriptionConfig;

/// High-level transcription function
pub async fn transcribe_audio_file<P: AsRef<std::path::Path>>(
    audio_path: P,
    config: Option<TranscriptionConfig>,
) -> Result<TranscriptionResult> {
    let config = config.unwrap_or_default();
    
    // Initialize transcriber
    let transcriber = WhisperTranscriber::new(config.clone()).await?;
    
    // Process audio
    let audio_processor = AudioProcessor::new();
    let audio_data = audio_processor.load_audio(audio_path).await?;
    
    // Transcribe
    transcriber.transcribe(audio_data).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_functionality() {
        // This test will be implemented once we have the modules ready
        assert!(true);
    }
}
