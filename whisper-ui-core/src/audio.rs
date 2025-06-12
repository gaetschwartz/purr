//! Audio processing functionality using FFmpeg

use crate::error::{Result, WhisperError};
use ffmpeg_next as ffmpeg;
use std::path::Path;
use tokio::task;

/// Audio data structure
#[derive(Debug, Clone)]
pub struct AudioData {
    /// Raw audio samples (f32, mono)
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration: f32,
}

/// Audio processor using FFmpeg
pub struct AudioProcessor {
    initialized: bool,
}

impl AudioProcessor {
    /// Create a new audio processor
    pub fn new() -> Self {
        Self {
            initialized: false,
        }
    }
    
    /// Initialize FFmpeg (call once)
    fn ensure_initialized(&mut self) -> Result<()> {
        if !self.initialized {
            ffmpeg::init().map_err(|e| WhisperError::FFmpeg(format!("Failed to initialize FFmpeg: {}", e)))?;
            self.initialized = true;
        }
        Ok(())
    }
    
    /// Load audio file and convert to the format expected by Whisper
    pub async fn load_audio<P: AsRef<Path>>(&mut self, path: P) -> Result<AudioData> {
        let path = path.as_ref().to_path_buf();
        
        // Run FFmpeg processing in a blocking task to avoid blocking the async runtime
        task::spawn_blocking(move || {
            let mut processor = AudioProcessor::new();
            processor.ensure_initialized()?;
            processor.load_audio_sync(&path)
        }).await
        .map_err(|e| WhisperError::AudioProcessing(format!("Task join error: {}", e)))?
    }
    
    /// Synchronous audio loading implementation
    fn load_audio_sync(&mut self, path: &Path) -> Result<AudioData> {
        self.ensure_initialized()?;
        
        // Validate file exists
        if !path.exists() {
            return Err(WhisperError::AudioProcessing(
                format!("Audio file not found: {}", path.display())
            ));
        }
        
        // Open input file
        let mut ictx = ffmpeg::format::input(&path)
            .map_err(|e| WhisperError::FFmpeg(format!("Failed to open audio file: {}", e)))?;
        
        // Find the audio stream
        let input = ictx
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .ok_or_else(|| WhisperError::AudioProcessing("No audio stream found".to_string()))?;
        
        let stream_index = input.index();
        
        // Get decoder
        let context_decoder = ffmpeg::codec::context::Context::from_parameters(input.parameters())
            .map_err(|e| WhisperError::FFmpeg(format!("Failed to create decoder context: {}", e)))?;
        
        let mut decoder = context_decoder
            .decoder()
            .audio()
            .map_err(|e| WhisperError::FFmpeg(format!("Failed to get audio decoder: {}", e)))?;
        
        // Setup resampler to convert to mono f32 at 16kHz (Whisper's preferred format)
        let mut resampler = ffmpeg::software::resampling::context::Context::get(
            decoder.format(),
            decoder.channel_layout(),
            decoder.rate(),
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Planar),
            ffmpeg::channel_layout::ChannelLayout::MONO,
            16000,
        ).map_err(|e| WhisperError::FFmpeg(format!("Failed to create resampler: {}", e)))?;
        
        let mut samples = Vec::new();
        let mut frame = ffmpeg::frame::Audio::empty();
        let mut resampled = ffmpeg::frame::Audio::empty();
        
        // Process packets
        for (stream, packet) in ictx.packets() {
            if stream.index() == stream_index {
                decoder.send_packet(&packet)
                    .map_err(|e| WhisperError::FFmpeg(format!("Failed to send packet: {}", e)))?;
                
                while decoder.receive_frame(&mut frame).is_ok() {
                    // Resample frame
                    resampler.run(&frame, &mut resampled)
                        .map_err(|e| WhisperError::FFmpeg(format!("Failed to resample: {}", e)))?;
                    
                    // Extract f32 samples
                    let data = resampled.data(0);
                    let sample_count = resampled.samples();
                    
                    unsafe {
                        let ptr = data.as_ptr() as *const f32;
                        let slice = std::slice::from_raw_parts(ptr, sample_count);
                        samples.extend_from_slice(slice);
                    }
                }
            }
        }
        
        // Flush decoder
        decoder.send_eof()
            .map_err(|e| WhisperError::FFmpeg(format!("Failed to flush decoder: {}", e)))?;
        
        while decoder.receive_frame(&mut frame).is_ok() {
            resampler.run(&frame, &mut resampled)
                .map_err(|e| WhisperError::FFmpeg(format!("Failed to resample final frames: {}", e)))?;
            
            let data = resampled.data(0);
            let sample_count = resampled.samples();
            
            unsafe {
                let ptr = data.as_ptr() as *const f32;
                let slice = std::slice::from_raw_parts(ptr, sample_count);
                samples.extend_from_slice(slice);
            }
        }
        
        let duration = samples.len() as f32 / 16000.0;
        
        Ok(AudioData {
            samples,
            sample_rate: 16000,
            duration,
        })
    }
}

impl Default for AudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_audio_processor_creation() {
        let processor = AudioProcessor::new();
        assert!(!processor.initialized);
    }
    
    #[tokio::test]
    async fn test_missing_file_error() {
        let mut processor = AudioProcessor::new();
        let result = processor.load_audio("nonexistent_file.wav").await;
        assert!(result.is_err());
    }
}
