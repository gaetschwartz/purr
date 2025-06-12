//! Simple audio processing without complex resampling

use crate::audio::AudioData;
use crate::error::{Result, WhisperError};
use std::path::Path;
use std::process::Command;
use tokio::task;

/// Simple audio processor that uses FFmpeg command line
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleAudioProcessor;

impl SimpleAudioProcessor {
    pub fn new() -> Self {
        Self
    }

    /// Load audio using ffmpeg command line to convert to the exact format we need
    pub async fn load_audio<P: AsRef<Path>>(&self, path: P) -> Result<AudioData> {
        let path = path.as_ref().to_path_buf();

        task::spawn_blocking(move || Self::load_audio_sync(&path))
            .await
            .map_err(|e| WhisperError::AudioProcessing(format!("Task join error: {}", e)))?
    }

    fn load_audio_sync(path: &Path) -> Result<AudioData> {
        // Validate file exists
        if !path.exists() {
            return Err(WhisperError::AudioProcessing(format!(
                "Audio file not found: {}",
                path.display()
            )));
        }

        // Use FFmpeg to convert to raw f32 samples at 16kHz mono
        let output = Command::new("ffmpeg")
            .args([
                "-loglevel",
                "quiet", // Suppress FFmpeg output
                "-i",
                &path.to_string_lossy(),
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "f32le",
                "-",
            ])
            .output()
            .map_err(|e| WhisperError::AudioProcessing(format!("Failed to run ffmpeg: {}", e)))?;

        if !output.status.success() {
            return Err(WhisperError::AudioProcessing(format!(
                "FFmpeg failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        // Convert raw bytes to f32 samples
        let raw_data = output.stdout;
        if raw_data.len() % 4 != 0 {
            return Err(WhisperError::AudioProcessing(
                "Invalid audio data length".to_string(),
            ));
        }

        let sample_count = raw_data.len() / 4;
        let mut samples = Vec::with_capacity(sample_count);

        for chunk in raw_data.chunks_exact(4) {
            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let sample = f32::from_le_bytes(bytes);
            samples.push(sample);
        }

        let duration = samples.len() as f32 / 16000.0;

        Ok(AudioData {
            samples,
            sample_rate: 16000,
            duration,
        })
    }
}
