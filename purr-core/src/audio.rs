//! Audio processing functionality using FFmpeg

use crate::error::{Result, WhisperError};
use ffmpeg_next as ffmpeg;
use std::path::Path;
use tokio::task;
use tracing::warn;

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
        Self { initialized: false }
    }

    /// Initialize FFmpeg (call once)
    fn ensure_initialized(&mut self) -> Result<()> {
        if !self.initialized {
            ffmpeg::init()
                .map_err(|e| WhisperError::FFmpeg(format!("Failed to initialize FFmpeg: {}", e)))?;

            // Set FFmpeg log level to quiet to suppress output
            unsafe {
                ffmpeg_next::sys::av_log_set_level(ffmpeg_next::sys::AV_LOG_QUIET);
            }

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
        })
        .await
        .map_err(|e| WhisperError::AudioProcessing(format!("Task join error: {}", e)))?
    }

    /// Synchronous audio loading implementation
    fn load_audio_sync(&mut self, path: &Path) -> Result<AudioData> {
        self.ensure_initialized()?;

        // Validate file exists
        if !path.exists() {
            return Err(WhisperError::AudioProcessing(format!(
                "Audio file not found: {}",
                path.display()
            )));
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
            .map_err(|e| {
                WhisperError::FFmpeg(format!("Failed to create decoder context: {}", e))
            })?;

        let mut decoder = context_decoder
            .decoder()
            .audio()
            .map_err(|e| WhisperError::FFmpeg(format!("Failed to get audio decoder: {}", e)))?;

        let mut samples = Vec::new();
        let mut frame = ffmpeg::frame::Audio::empty();
        let mut resampled = ffmpeg::frame::Audio::empty();

        // Resampler state tracking
        let mut resampler: Option<ffmpeg::software::resampling::context::Context> = None;
        let mut last_format: Option<ffmpeg::format::Sample> = None;
        let mut last_channel_layout: Option<ffmpeg::channel_layout::ChannelLayout> = None;
        let mut last_rate: Option<u32> = None;

        // Process packets with error resilience
        for (stream, packet) in ictx.packets() {
            if stream.index() == stream_index {
                // Try to send packet, but continue on errors to handle corrupted streams
                match decoder.send_packet(&packet) {
                    Ok(()) => {
                        // Successfully sent packet, process frames
                        while decoder.receive_frame(&mut frame).is_ok() {
                            Self::process_audio_frame(
                                &frame,
                                &mut samples,
                                &mut resampled,
                                &mut resampler,
                                &mut last_format,
                                &mut last_channel_layout,
                                &mut last_rate,
                            )?;
                        }
                    }
                    Err(ffmpeg_next::Error::InvalidData) => {
                        // Log the error but continue processing - skip corrupted packets
                        warn!("Skipping invalid chunk at stream index {}", stream_index,);
                        continue;
                    }
                    Err(e) => {
                        return Err(WhisperError::FFmpeg(format!(
                            "Failed to send packet to decoder: {}",
                            e
                        )))
                    }
                }
            }
        }

        // Flush decoder - continue even if flushing fails
        match decoder.send_eof() {
            Ok(()) => {
                while decoder.receive_frame(&mut frame).is_ok() {
                    Self::process_audio_frame(
                        &frame,
                        &mut samples,
                        &mut resampled,
                        &mut resampler,
                        &mut last_format,
                        &mut last_channel_layout,
                        &mut last_rate,
                    )?;
                }
            }
            Err(e) => {
                eprintln!("Warning: Failed to flush decoder, but continuing: {}", e);
            }
        }

        // Check if we got any audio data
        if samples.is_empty() {
            return Err(WhisperError::AudioProcessing(
                "No audio data could be extracted from file - file may be corrupted or unsupported"
                    .to_string(),
            ));
        }

        let duration = samples.len() as f32 / 16000.0;

        Ok(AudioData {
            samples,
            sample_rate: 16000,
            duration,
        })
    }

    /// Process a single audio frame with proper resampling
    fn process_audio_frame(
        frame: &ffmpeg::frame::Audio,
        samples: &mut Vec<f32>,
        resampled: &mut ffmpeg::frame::Audio,
        resampler: &mut Option<ffmpeg::software::resampling::context::Context>,
        last_format: &mut Option<ffmpeg::format::Sample>,
        last_channel_layout: &mut Option<ffmpeg::channel_layout::ChannelLayout>,
        last_rate: &mut Option<u32>,
    ) -> Result<()> {
        // Check if frame properties have changed and we need to recreate the resampler
        let current_format = frame.format();
        let current_rate = frame.rate();

        // Determine the channel layout for this frame
        let current_channel_layout = if frame.channel_layout().channels() == 0 {
            // Use default based on channel count
            match frame.channels() {
                1 => ffmpeg::channel_layout::ChannelLayout::MONO,
                2 => ffmpeg::channel_layout::ChannelLayout::STEREO,
                _ => ffmpeg::channel_layout::ChannelLayout::default(frame.channels() as i32),
            }
        } else {
            frame.channel_layout()
        };

        // Check if we need to recreate the resampler
        let resampler_needs_update = (*last_format != Some(current_format))
            || (*last_channel_layout != Some(current_channel_layout))
            || (*last_rate != Some(current_rate));

        // Determine if we need resampling at all
        // We can do direct conversion for I16 and F32 formats for mono audio
        let is_direct_convertible = matches!(
            current_format,
            ffmpeg::format::Sample::I16(_) | ffmpeg::format::Sample::F32(_)
        ) && frame.channels() == 1;

        // We can handle direct conversion even for different sample rates
        let needs_resampling = !is_direct_convertible;

        if needs_resampling {
            // Create or recreate resampler if needed
            if resampler_needs_update || resampler.is_none() {
                *resampler = Some(
                    ffmpeg::software::resampling::context::Context::get(
                        current_format,
                        current_channel_layout,
                        current_rate,
                        ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Planar),
                        ffmpeg::channel_layout::ChannelLayout::MONO,
                        16000,
                    )
                    .map_err(|e| {
                        WhisperError::FFmpeg(format!("Failed to create resampler: {}", e))
                    })?,
                );

                // Update our tracking variables
                *last_format = Some(current_format);
                *last_channel_layout = Some(current_channel_layout);
                *last_rate = Some(current_rate);
            }

            // Resample frame with error handling
            if let Some(ref mut resampler_ctx) = resampler {
                match resampler_ctx.run(frame, resampled) {
                    Ok(_) => {
                        // Successfully resampled
                    }
                    Err(e) => {
                        // Input format changed - skip this frame and continue
                        eprintln!("Warning: Skipping frame due to resampling error: {}", e);

                        // Force recreation of resampler for next frame
                        *resampler = None;
                        *last_format = None;
                        *last_channel_layout = None;
                        *last_rate = None;

                        // Skip processing this frame but continue with next ones
                        return Ok(());
                    }
                }
            }

            // Extract f32 samples
            let data = resampled.data(0);
            let sample_count = resampled.samples();

            unsafe {
                let ptr = data.as_ptr() as *const f32;
                let slice = std::slice::from_raw_parts(ptr, sample_count);
                samples.extend_from_slice(slice);
            }
        } else {
            // Direct conversion for mono audio with manual sample rate conversion
            let data = frame.data(0);
            let sample_count = frame.samples();

            match current_format {
                ffmpeg::format::Sample::I16(_) => {
                    // Convert s16 to f32 with potential downsampling
                    unsafe {
                        let ptr = data.as_ptr() as *const i16;
                        let slice = std::slice::from_raw_parts(ptr, sample_count);

                        if current_rate == 16000 {
                            // Direct conversion for 16kHz
                            for &sample in slice {
                                samples.push(sample as f32 / 32768.0);
                            }
                        } else {
                            // Simple downsampling for other rates
                            let step = current_rate as f32 / 16000.0;
                            let mut pos = 0.0;
                            while (pos as usize) < slice.len() {
                                let idx = pos as usize;
                                samples.push(slice[idx] as f32 / 32768.0);
                                pos += step;
                            }
                        }
                    }
                }
                ffmpeg::format::Sample::F32(_) => {
                    // F32 conversion with potential downsampling
                    unsafe {
                        let ptr = data.as_ptr() as *const f32;
                        let slice = std::slice::from_raw_parts(ptr, sample_count);

                        if current_rate == 16000 {
                            // Direct copy for 16kHz
                            samples.extend_from_slice(slice);
                        } else {
                            // Simple downsampling for other rates
                            let step = current_rate as f32 / 16000.0;
                            let mut pos = 0.0;
                            while (pos as usize) < slice.len() {
                                let idx = pos as usize;
                                samples.push(slice[idx]);
                                pos += step;
                            }
                        }
                    }
                }
                _ => {
                    // This shouldn't happen since we checked needs_resampling above
                    return Err(WhisperError::AudioProcessing(format!(
                        "Unexpected audio format in direct conversion path: {:?}",
                        current_format
                    )));
                }
            }
        }

        Ok(())
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
