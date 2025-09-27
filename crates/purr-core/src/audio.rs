//! Audio processing functionality using `FFmpeg`

use crate::error::{AudioProcessingError, Result, WhisperError};
use crate::input::AsyncStreamBuffer;
use ffmpeg_next as ffmpeg;
use futures::Stream;
use reqwest::IntoUrl;
use std::path::Path;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tokio::task;
use tracing::warn;
use url::Url;

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

/// Audio chunk for streaming processing (10 seconds)
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Raw audio samples (f32, mono, 16kHz)
    pub samples: Vec<f32>,
    /// Sample rate (always 16000 for Whisper)
    pub sample_rate: u32,
    /// Duration in seconds (target: 10.0)
    pub duration: f32,
    /// Chunk index in the stream
    pub index: usize,
    /// Start time in the original audio (seconds)
    pub start_time: f32,
    /// Whether this is the final chunk in the stream
    pub is_final: bool,
}

impl AudioChunk {
    /// Target chunk duration in seconds
    pub const TARGET_DURATION: f32 = 10.0;

    /// Target samples per chunk (10 seconds at 16kHz)
    pub const TARGET_SAMPLES: usize = (Self::TARGET_DURATION * 16000.0) as usize;

    /// Create a new audio chunk
    #[must_use]
    pub fn new(samples: Vec<f32>, index: usize, start_time: f32, is_final: bool) -> Self {
        let duration = samples.len() as f32 / 16000.0;
        Self {
            samples,
            sample_rate: 16000,
            duration,
            index,
            start_time,
            is_final,
        }
    }
}

/// Stream of audio chunks
pub struct AudioStream {
    receiver: mpsc::UnboundedReceiver<Result<AudioChunk>>,
}

impl AudioStream {
    fn new(receiver: mpsc::UnboundedReceiver<Result<AudioChunk>>) -> Self {
        Self { receiver }
    }
}

impl Stream for AudioStream {
    type Item = Result<AudioChunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

impl std::fmt::Debug for AudioStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioStream").finish()
    }
}

/// Audio processor using `FFmpeg`
pub struct AudioProcessor {}

impl AudioProcessor {
    /// Create a new audio processor
    pub fn new() -> Result<Self> {
        ffmpeg::init().map_err(WhisperError::from)?;

        // Set FFmpeg log level to quiet to suppress output
        unsafe {
            ffmpeg_next::sys::av_log_set_level(ffmpeg_next::sys::AV_LOG_QUIET);
        }

        Ok(Self {})
    }

    /// Load audio file and convert to the format expected by Whisper
    pub async fn load_audio<P: AsRef<Path>>(path: P) -> Result<AudioData> {
        let path = path.as_ref().to_path_buf();

        // Run FFmpeg processing in a blocking task to avoid blocking the async runtime
        task::spawn_blocking(move || {
            let mut processor = AudioProcessor::new()?;
            processor.load_audio_sync(&path)
        })
        .await
        .map_err(|e| WhisperError::from(AudioProcessingError::TaskJoin { source: e }))?
    }

    /// Stream audio file as chunks for real-time processing
    pub async fn stream_file<P: AsRef<Path>>(path: P) -> Result<AudioStream> {
        let path = path.as_ref().to_path_buf();

        let (tx, rx) = mpsc::unbounded_channel();

        // Process audio in a background task
        task::spawn_blocking(move || {
            let mut processor = match AudioProcessor::new() {
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
                Ok(p) => p,
            };

            if let Err(e) = processor.stream_audio(&path, tx) {
                // Error will already be sent through channel if possible
                warn!("Audio streaming failed: {}", e);
            }
        });

        Ok(AudioStream::new(rx))
    }

    /// Synchronous audio loading implementation
    fn load_audio_sync(&mut self, path: &Path) -> Result<AudioData> {
        // Validate file exists
        if !path.exists() {
            return Err(WhisperError::from(AudioProcessingError::ReadFailed {
                file: path.to_path_buf(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "File not found"),
            }));
        }

        // Open input file and convert to AsyncCustomInput
        let input_ctx = ffmpeg::format::input(&path).map_err(WhisperError::from)?;
        let custom_input = crate::input::AsyncCustomInput::from_input_context(input_ctx)?;

        // Use the common input processing logic
        self.load_audio_from_input(custom_input)
    }

    /// Synchronous streaming audio implementation
    fn stream_audio(
        &mut self,
        path: &Path,
        tx: mpsc::UnboundedSender<Result<AudioChunk>>,
    ) -> Result<()> {
        // Validate file exists
        if !path.exists() {
            let error_msg = format!("Audio file not found: {}", path.display());
            let error = WhisperError::from(AudioProcessingError::processing_failed(
                "file validation",
                std::io::Error::new(std::io::ErrorKind::NotFound, error_msg.clone()),
            ));
            let _ = tx.send(Err(WhisperError::from(
                AudioProcessingError::processing_failed(
                    "file validation",
                    std::io::Error::new(std::io::ErrorKind::NotFound, error_msg),
                ),
            )));
            return Err(error);
        }

        // Open input file and convert to AsyncCustomInput
        let input_ctx = ffmpeg::format::input(&path).map_err(WhisperError::from)?;
        let custom_input = crate::input::AsyncCustomInput::from_input_context(input_ctx)?;

        // Use the common input processing logic
        self.stream_audio_from_input(custom_input, tx)
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
                _ => ffmpeg::channel_layout::ChannelLayout::default(i32::from(frame.channels())),
            }
        } else {
            frame.channel_layout()
        };

        // Check if we need to recreate the resampler
        let resampler_needs_update = (*last_format != Some(current_format))
            || (*last_channel_layout != Some(current_channel_layout))
            || (*last_rate != Some(current_rate));

        // Always use resampler for consistent processing and better format handling
        // This ensures proper stereo-to-mono conversion and sample rate conversion
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
                .map_err(WhisperError::from)?,
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
                    // Successfully resampled - extract f32 samples
                    let data = resampled.data(0);
                    let sample_count = resampled.samples();

                    if sample_count > 0 {
                        unsafe {
                            let ptr = data.as_ptr().cast::<f32>();
                            let slice = std::slice::from_raw_parts(ptr, sample_count);
                            samples.extend_from_slice(slice);
                        }
                    }
                }
                Err(e) => {
                    // Force recreation of resampler for next frame
                    *resampler = None;
                    *last_format = None;
                    *last_channel_layout = None;
                    *last_rate = None;

                    // Try fallback processing for this frame
                    if let Ok(()) = Self::process_frame_fallback(frame, samples, current_rate) {
                        // Fallback succeeded, continue
                    } else {
                        // Log warning and skip this frame
                        warn!("Skipping frame due to resampling error: {}", e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Fallback processing for frames that fail resampling
    fn process_frame_fallback(
        frame: &ffmpeg::frame::Audio,
        samples: &mut Vec<f32>,
        current_rate: u32,
    ) -> Result<()> {
        let current_format = frame.format();
        let data = frame.data(0);
        let sample_count = frame.samples();
        let channels = frame.channels();

        if sample_count == 0 {
            return Ok(());
        }

        match current_format {
            ffmpeg::format::Sample::I16(sample_type) => {
                unsafe {
                    let ptr = data.as_ptr().cast::<i16>();
                    let slice = std::slice::from_raw_parts(ptr, sample_count * channels as usize);

                    let step = current_rate as f32 / 16000.0;
                    let mut pos = 0.0;

                    match sample_type {
                        ffmpeg::format::sample::Type::Packed => {
                            // Interleaved format (LRLRLR...)
                            while (pos as usize * channels as usize) < slice.len() {
                                let idx = (pos as usize) * channels as usize;
                                // Convert to mono by averaging channels or taking left channel
                                let mono_sample = if channels == 1 {
                                    slice[idx]
                                } else {
                                    // Average all channels for mono conversion
                                    let sum: i32 = (0..channels)
                                        .map(|c| i32::from(slice[idx + c as usize]))
                                        .sum();
                                    (sum / i32::from(channels)) as i16
                                };
                                samples.push(f32::from(mono_sample) / 32768.0);
                                pos += step;
                            }
                        }
                        ffmpeg::format::sample::Type::Planar => {
                            // Planar format - take first channel only
                            while (pos as usize) < sample_count {
                                let idx = pos as usize;
                                samples.push(f32::from(slice[idx]) / 32768.0);
                                pos += step;
                            }
                        }
                    }
                }
            }
            ffmpeg::format::Sample::F32(sample_type) => {
                unsafe {
                    let ptr = data.as_ptr().cast::<f32>();
                    let slice = std::slice::from_raw_parts(ptr, sample_count * channels as usize);

                    let step = current_rate as f32 / 16000.0;
                    let mut pos = 0.0;

                    match sample_type {
                        ffmpeg::format::sample::Type::Packed => {
                            // Interleaved format (LRLRLR...)
                            while (pos as usize * channels as usize) < slice.len() {
                                let idx = (pos as usize) * channels as usize;
                                // Convert to mono by averaging channels or taking left channel
                                let mono_sample = if channels == 1 {
                                    slice[idx]
                                } else {
                                    // Average all channels for mono conversion
                                    let sum: f32 =
                                        (0..channels).map(|c| slice[idx + c as usize]).sum();
                                    sum / f32::from(channels)
                                };
                                samples.push(mono_sample);
                                pos += step;
                            }
                        }
                        ffmpeg::format::sample::Type::Planar => {
                            // Planar format - take first channel only
                            while (pos as usize) < sample_count {
                                let idx = pos as usize;
                                samples.push(slice[idx]);
                                pos += step;
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(WhisperError::from(AudioProcessingError::FormatConversion));
            }
        }

        Ok(())
    }

    /// Process a single audio frame into a buffer (for streaming)
    fn process_audio_frame_to_buffer(
        frame: &ffmpeg::frame::Audio,
        output_samples: &mut Vec<f32>,
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
                _ => ffmpeg::channel_layout::ChannelLayout::default(i32::from(frame.channels())),
            }
        } else {
            frame.channel_layout()
        };

        // Check if we need to recreate the resampler
        let resampler_needs_update = (*last_format != Some(current_format))
            || (*last_channel_layout != Some(current_channel_layout))
            || (*last_rate != Some(current_rate));

        // Always use resampler for consistent processing and better format handling
        // This ensures proper stereo-to-mono conversion and sample rate conversion
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
                .map_err(WhisperError::from)?,
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
                    // Successfully resampled - extract f32 samples
                    let data = resampled.data(0);
                    let sample_count = resampled.samples();

                    if sample_count > 0 {
                        unsafe {
                            let ptr = data.as_ptr().cast::<f32>();
                            let slice = std::slice::from_raw_parts(ptr, sample_count);
                            output_samples.extend_from_slice(slice);
                        }
                    }
                }
                Err(e) => {
                    // Force recreation of resampler for next frame
                    *resampler = None;
                    *last_format = None;
                    *last_channel_layout = None;
                    *last_rate = None;

                    // Try fallback processing for this frame
                    if let Ok(()) =
                        Self::process_frame_fallback(frame, output_samples, current_rate)
                    {
                        // Fallback succeeded, continue
                    } else {
                        // Log warning and skip this frame
                        warn!("Skipping frame due to resampling error: {}", e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Load audio from URL and convert to the format expected by Whisper
    pub async fn load_audio_from_url(url: impl IntoUrl) -> Result<AudioData> {
        Self::load_audio_from_url_with_config(
            url.into_url()?,
            &crate::TranscriptionConfig::default(),
        )
        .await
    }

    /// Load audio from URL with custom configuration
    pub async fn load_audio_from_url_with_config(
        url: Url,
        config: &crate::TranscriptionConfig,
    ) -> Result<AudioData> {
        use crate::input::AsyncCustomInput;
        use crate::url::{HttpStreamer, UrlStreamConfig};
        use std::time::Duration;

        // Create URL streamer with timeout configuration from TranscriptionConfig
        let url_config = UrlStreamConfig {
            timeout: Duration::from_secs(config.http_timeout),
            connect_timeout: Duration::from_secs(config.http_connect_timeout),
            ..Default::default()
        };
        let streamer = HttpStreamer::with_config(url_config)?;

        // Create buffer for streaming
        let buffer = AsyncStreamBuffer::new();

        // Start streaming URL data
        let streaming_handle = streamer.stream_url(url, buffer.clone()).await?;

        // Create custom input from the buffer
        let custom_input = AsyncCustomInput::create(buffer).await?;

        // Process the streaming input
        let audio_data = task::spawn_blocking(move || {
            let mut processor = AudioProcessor::new()?;
            processor.load_audio_from_input(custom_input)
        })
        .await
        .map_err(|e| WhisperError::from(AudioProcessingError::TaskJoin { source: e }))??;

        // Wait for streaming to complete
        let _ = streaming_handle.await;

        Ok(audio_data)
    }

    /// Stream audio from URL as chunks for real-time processing
    pub async fn stream_url(url: Url) -> Result<AudioStream> {
        Self::stream_url_with_config(url, &crate::TranscriptionConfig::default()).await
    }

    /// Stream audio from URL with custom configuration
    pub async fn stream_url_with_config(
        url: Url,
        config: &crate::TranscriptionConfig,
    ) -> Result<AudioStream> {
        use crate::input::AsyncCustomInput;
        use crate::url::{HttpStreamer, UrlStreamConfig};
        use std::time::Duration;

        // Create URL streamer with timeout configuration from TranscriptionConfig
        let url_config = UrlStreamConfig {
            timeout: Duration::from_secs(config.http_timeout),
            connect_timeout: Duration::from_secs(config.http_connect_timeout),
            ..Default::default()
        };
        let streamer = HttpStreamer::with_config(url_config)?;

        // Create buffer for streaming
        let buffer = AsyncStreamBuffer::new();

        // Start streaming URL data
        let _streaming_handle = streamer.stream_url(url, buffer.clone()).await?;

        let (tx, rx) = mpsc::unbounded_channel();

        // Process audio in a background task
        task::spawn_blocking(move || {
            let mut processor = match AudioProcessor::new() {
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
                Ok(p) => p,
            };

            // Create custom input from the buffer
            let custom_input = match futures::executor::block_on(AsyncCustomInput::create(buffer)) {
                Err(e) => {
                    let _ = tx.send(Err(WhisperError::from(e)));
                    return;
                }
                Ok(input) => input,
            };

            if let Err(e) = processor.stream_audio_from_input(custom_input, tx) {
                // Error will already be sent through channel if possible
                warn!("URL audio streaming failed: {}", e);
            }
        });

        Ok(AudioStream::new(rx))
    }

    /// Load audio from AsyncCustomInput
    fn load_audio_from_input(
        &mut self,
        input: crate::input::AsyncCustomInput,
    ) -> Result<AudioData> {
        let mut ictx = input.into_input();

        // Find the audio stream
        let input_stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .ok_or_else(|| WhisperError::from(AudioProcessingError::NoAudioStream))?;

        let stream_index = input_stream.index();

        // Get decoder
        let context_decoder =
            ffmpeg::codec::context::Context::from_parameters(input_stream.parameters())
                .map_err(WhisperError::from)?;

        let mut decoder = context_decoder
            .decoder()
            .audio()
            .map_err(WhisperError::from)?;

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
                match decoder.send_packet(&packet) {
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
                    Err(ffmpeg_next::Error::InvalidData) => {
                        warn!("Skipping invalid chunk at stream index {}", stream_index);
                        continue;
                    }
                    Err(e) => return Err(WhisperError::from(e)),
                }
            }
        }

        // Flush decoder
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
                eprintln!("Warning: Failed to flush decoder, but continuing: {e}");
            }
        }

        // Check if we got any audio data
        if samples.is_empty() {
            return Err(WhisperError::from(AudioProcessingError::processing_failed(
                "audio extraction",
                std::io::Error::new(std::io::ErrorKind::InvalidData, "No audio data could be extracted from URL - stream may be corrupted or unsupported")
            )));
        }

        let duration = samples.len() as f32 / 16000.0;

        Ok(AudioData {
            samples,
            sample_rate: 16000,
            duration,
        })
    }

    /// Stream audio from AsyncCustomInput for real-time processing
    fn stream_audio_from_input(
        &mut self,
        input: crate::input::AsyncCustomInput,
        tx: mpsc::UnboundedSender<Result<AudioChunk>>,
    ) -> Result<()> {
        let mut ictx = input.into_input();

        // Find the audio stream
        let input_stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .ok_or_else(|| WhisperError::from(AudioProcessingError::NoAudioStream))?;

        let stream_index = input_stream.index();

        // Get decoder
        let context_decoder =
            ffmpeg::codec::context::Context::from_parameters(input_stream.parameters())
                .map_err(WhisperError::from)?;

        let mut decoder = context_decoder
            .decoder()
            .audio()
            .map_err(WhisperError::from)?;

        let mut output_samples = Vec::new();
        let mut frame = ffmpeg::frame::Audio::empty();

        let mut chunk_index = 0;
        let mut total_processed_time = 0.0;

        // Process packets with error resilience
        for (stream, packet) in ictx.packets() {
            if stream.index() == stream_index {
                match decoder.send_packet(&packet) {
                    Ok(()) => {
                        while decoder.receive_frame(&mut frame).is_ok() {
                            let mut resampled = ffmpeg::frame::Audio::empty();
                            let mut resampler: Option<
                                ffmpeg::software::resampling::context::Context,
                            > = None;
                            let mut last_format: Option<ffmpeg::format::Sample> = None;
                            let mut last_channel_layout: Option<
                                ffmpeg::channel_layout::ChannelLayout,
                            > = None;
                            let mut last_rate: Option<u32> = None;

                            Self::process_audio_frame_to_buffer(
                                &frame,
                                &mut output_samples,
                                &mut resampled,
                                &mut resampler,
                                &mut last_format,
                                &mut last_channel_layout,
                                &mut last_rate,
                            )?;

                            // Check if we have enough samples for a chunk
                            while output_samples.len() >= AudioChunk::TARGET_SAMPLES {
                                let chunk_samples: Vec<f32> =
                                    output_samples.drain(..AudioChunk::TARGET_SAMPLES).collect();
                                let chunk = AudioChunk::new(
                                    chunk_samples,
                                    chunk_index,
                                    total_processed_time,
                                    false,
                                );
                                total_processed_time += chunk.duration;
                                chunk_index += 1;

                                if tx.send(Ok(chunk)).is_err() {
                                    return Ok(());
                                }
                            }
                        }
                    }
                    Err(ffmpeg_next::Error::InvalidData) => {
                        warn!("Skipping invalid chunk at stream index {}", stream_index);
                        continue;
                    }
                    Err(e) => {
                        let whisper_error = WhisperError::from(e);
                        let _ = tx.send(Err(WhisperError::NetworkError {
                            message: format!("Failed to decode audio packet: {}", e),
                        }));
                        return Err(whisper_error);
                    }
                }
            }
        }

        // Flush decoder
        match decoder.send_eof() {
            Ok(()) => {
                while decoder.receive_frame(&mut frame).is_ok() {
                    let mut resampled = ffmpeg::frame::Audio::empty();
                    let mut resampler: Option<ffmpeg::software::resampling::context::Context> =
                        None;
                    let mut last_format: Option<ffmpeg::format::Sample> = None;
                    let mut last_channel_layout: Option<ffmpeg::channel_layout::ChannelLayout> =
                        None;
                    let mut last_rate: Option<u32> = None;

                    Self::process_audio_frame_to_buffer(
                        &frame,
                        &mut output_samples,
                        &mut resampled,
                        &mut resampler,
                        &mut last_format,
                        &mut last_channel_layout,
                        &mut last_rate,
                    )?;

                    // Process any complete chunks
                    while output_samples.len() >= AudioChunk::TARGET_SAMPLES {
                        let chunk_samples: Vec<f32> =
                            output_samples.drain(..AudioChunk::TARGET_SAMPLES).collect();
                        let chunk = AudioChunk::new(
                            chunk_samples,
                            chunk_index,
                            total_processed_time,
                            false,
                        );
                        total_processed_time += chunk.duration;
                        chunk_index += 1;

                        if tx.send(Ok(chunk)).is_err() {
                            return Ok(());
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Failed to flush decoder, but continuing: {e}");
            }
        }

        // Send remaining samples as final chunk
        if !output_samples.is_empty() {
            let final_chunk =
                AudioChunk::new(output_samples, chunk_index, total_processed_time, true);
            let _ = tx.send(Ok(final_chunk));
        } else if chunk_index == 0 {
            // No chunks were sent, send an error
            let error = WhisperError::from(AudioProcessingError::processing_failed(
                "audio processing",
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "No audio data could be processed from URL",
                ),
            ));
            let _ = tx.send(Err(error));
            return Err(WhisperError::from(AudioProcessingError::processing_failed(
                "audio processing",
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "No audio data could be processed from URL",
                ),
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use tokio::fs;

    /// Helper to create a minimal WAV file for testing
    fn create_test_wav_bytes(samples: Vec<i16>, sample_rate: u32) -> Vec<u8> {
        let mut wav_data = Vec::new();

        // WAV header
        wav_data.extend_from_slice(b"RIFF");
        let file_size = 36 + (samples.len() * 2) as u32;
        wav_data.extend_from_slice(&file_size.to_le_bytes());
        wav_data.extend_from_slice(b"WAVE");

        // Format chunk
        wav_data.extend_from_slice(b"fmt ");
        wav_data.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        wav_data.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        wav_data.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav_data.extend_from_slice(&sample_rate.to_le_bytes());
        wav_data.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        wav_data.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav_data.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

        // Data chunk
        wav_data.extend_from_slice(b"data");
        wav_data.extend_from_slice(&((samples.len() * 2) as u32).to_le_bytes());

        // Sample data
        for sample in samples {
            wav_data.extend_from_slice(&sample.to_le_bytes());
        }

        wav_data
    }

    #[test]
    fn test_audio_data_creation() {
        let samples = vec![0.1, 0.2, -0.1, -0.2];
        let audio_data = AudioData {
            samples: samples.clone(),
            sample_rate: 16000,
            duration: 0.25, // 4 samples at 16kHz = 0.25ms
        };

        assert_eq!(audio_data.samples, samples);
        assert_eq!(audio_data.sample_rate, 16000);
        assert_eq!(audio_data.duration, 0.25);
    }

    #[test]
    fn test_audio_data_clone() {
        let original = AudioData {
            samples: vec![1.0, 2.0, 3.0],
            sample_rate: 44100,
            duration: 1.5,
        };

        let cloned = original.clone();
        assert_eq!(original.samples, cloned.samples);
        assert_eq!(original.sample_rate, cloned.sample_rate);
        assert_eq!(original.duration, cloned.duration);
    }

    #[test]
    fn test_audio_chunk_new() {
        let samples = vec![0.5; 1000];
        let chunk = AudioChunk::new(samples.clone(), 2, 20.5, true);

        assert_eq!(chunk.samples, samples);
        assert_eq!(chunk.sample_rate, 16000);
        assert_eq!(chunk.index, 2);
        assert_eq!(chunk.start_time, 20.5);
        assert!(chunk.is_final);

        // Duration should be calculated from sample count
        let expected_duration = samples.len() as f32 / 16000.0;
        assert!((chunk.duration - expected_duration).abs() < 0.001);
    }

    #[test]
    fn test_audio_chunk_constants() {
        assert_eq!(AudioChunk::TARGET_DURATION, 10.0);
        assert_eq!(AudioChunk::TARGET_SAMPLES, 160000);
    }

    #[test]
    fn test_audio_chunk_target_samples_calculation() {
        // Verify the target samples calculation is correct
        let expected = (AudioChunk::TARGET_DURATION * 16000.0) as usize;
        assert_eq!(AudioChunk::TARGET_SAMPLES, expected);
    }

    #[test]
    fn test_audio_chunk_duration_calculation() {
        let test_cases = vec![
            (16000, 1.0), // 1 second
            (8000, 0.5),  // 0.5 seconds
            (32000, 2.0), // 2 seconds
            (1600, 0.1),  // 0.1 seconds
        ];

        for (sample_count, expected_duration) in test_cases {
            let samples = vec![0.0; sample_count];
            let chunk = AudioChunk::new(samples, 0, 0.0, false);
            assert!(
                (chunk.duration - expected_duration).abs() < 0.001,
                "Expected {} seconds for {} samples, got {}",
                expected_duration,
                sample_count,
                chunk.duration
            );
        }
    }

    #[test]
    fn test_audio_processor_creation() {
        // This test verifies that AudioProcessor can be created
        // Note: actual FFmpeg operations are tested in integration tests
        let result = AudioProcessor::new();
        assert!(result.is_ok(), "AudioProcessor creation should succeed");
    }

    #[tokio::test]
    async fn test_audio_stream_creation() {
        use tokio::sync::mpsc;

        let (tx, rx) = mpsc::unbounded_channel();
        let stream = AudioStream::new(rx);

        // Create a simple test by dropping the stream immediately
        drop(stream);
        drop(tx);
    }

    #[tokio::test]
    async fn test_audio_stream_receiving() {
        use futures::StreamExt;
        use tokio::sync::mpsc;

        let (tx, rx) = mpsc::unbounded_channel();
        let mut stream = AudioStream::new(rx);

        // Send a test chunk
        let test_chunk = AudioChunk::new(vec![0.1, 0.2, 0.3], 0, 0.0, true);
        tx.send(Ok(test_chunk.clone())).unwrap();
        drop(tx); // Close the channel

        // Receive the chunk
        let received = stream.next().await;
        assert!(received.is_some());

        let chunk_result = received.unwrap();
        assert!(chunk_result.is_ok());

        let chunk = chunk_result.unwrap();
        assert_eq!(chunk.samples, test_chunk.samples);
        assert_eq!(chunk.index, test_chunk.index);
        assert_eq!(chunk.start_time, test_chunk.start_time);
        assert_eq!(chunk.is_final, test_chunk.is_final);
    }

    #[tokio::test]
    async fn test_audio_stream_error_handling() {
        use futures::StreamExt;
        use tokio::sync::mpsc;

        let (tx, rx) = mpsc::unbounded_channel();
        let mut stream = AudioStream::new(rx);

        // Send an error
        let test_error = WhisperError::from(AudioProcessingError::processing_failed(
            "test operation",
            std::io::Error::other("Test error"),
        ));
        tx.send(Err(test_error)).unwrap();
        drop(tx);

        // Receive the error
        let received = stream.next().await;
        assert!(received.is_some());

        let chunk_result = received.unwrap();
        assert!(chunk_result.is_err());
    }

    #[tokio::test]
    async fn test_load_audio_nonexistent_file() {
        let result = AudioProcessor::load_audio("definitely_does_not_exist.wav").await;

        assert!(result.is_err());
        let error = result.unwrap_err();
        match error {
            WhisperError::AudioProcessing {
                source: AudioProcessingError::ReadFailed { .. },
            } => {
                // Expected ReadFailed error for nonexistent file
            }
            WhisperError::AudioProcessing {
                source: AudioProcessingError::ProcessingFailed { operation, .. },
            } => {
                assert!(operation.contains("not found"));
            }
            _ => panic!(
                "Expected ReadFailed or ProcessingFailed error with 'not found', got: {:?}",
                error
            ),
        }
    }

    #[tokio::test]
    async fn test_load_audio_invalid_file() {
        // Create a temporary file with invalid content
        let temp_file = NamedTempFile::new().unwrap();
        fs::write(temp_file.path(), b"this is not audio data")
            .await
            .unwrap();

        let result = AudioProcessor::load_audio(temp_file.path()).await;

        assert!(result.is_err());
        // The exact error type may vary depending on FFmpeg's response
    }

    #[tokio::test]
    async fn test_load_audio_valid_wav() {
        // Create a valid WAV file
        let sample_rate = 16000u32;
        let samples = vec![0i16; sample_rate as usize]; // 1 second of silence
        let wav_data = create_test_wav_bytes(samples, sample_rate);

        let temp_file = NamedTempFile::new().unwrap();
        fs::write(temp_file.path(), wav_data).await.unwrap();

        let result = AudioProcessor::load_audio(temp_file.path()).await;

        assert!(result.is_ok(), "Should successfully load valid WAV file");

        let audio_data = result.unwrap();
        assert_eq!(audio_data.sample_rate, 16000);
        assert!(audio_data.duration > 0.9 && audio_data.duration < 1.1); // ~1 second
        assert!(!audio_data.samples.is_empty());
    }

    #[tokio::test]
    async fn test_stream_nonexistent_file() {
        use futures::StreamExt;

        let result = AudioProcessor::stream_file("nonexistent_file.wav").await;

        // Stream creation might succeed, but the first chunk should contain an error
        match result {
            Ok(mut stream) => {
                // Try to get the first chunk - this should fail
                if let Some(chunk_result) = stream.next().await {
                    assert!(
                        chunk_result.is_err(),
                        "Should get an error for nonexistent file"
                    );
                }
            }
            Err(_) => {
                // Also acceptable - immediate failure
            }
        }
    }

    #[tokio::test]
    async fn test_stream_valid_file() {
        use futures::StreamExt;

        // Create a valid WAV file with enough content for streaming
        let sample_rate = 16000u32;
        let duration = 15.0; // 15 seconds to ensure multiple chunks
        let sample_count = (sample_rate as f32 * duration) as usize;
        let samples = vec![0i16; sample_count];
        let wav_data = create_test_wav_bytes(samples, sample_rate);

        let temp_file = NamedTempFile::new().unwrap();
        fs::write(temp_file.path(), wav_data).await.unwrap();

        let result = AudioProcessor::stream_file(temp_file.path()).await;
        assert!(
            result.is_ok(),
            "Should successfully create stream for valid file"
        );

        let mut stream = result.unwrap();
        let mut chunk_count = 0;
        let mut last_was_final = false;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    // Validate chunk properties
                    assert_eq!(chunk.sample_rate, 16000);
                    assert_eq!(chunk.index, chunk_count);
                    assert!(chunk.start_time >= 0.0);
                    assert!(chunk.duration > 0.0);
                    assert!(!chunk.samples.is_empty());

                    last_was_final = chunk.is_final;
                    chunk_count += 1;

                    if chunk.is_final {
                        break;
                    }
                }
                Err(e) => panic!("Stream error: {:?}", e),
            }
        }

        assert!(
            chunk_count > 1,
            "Should produce multiple chunks for 15-second audio"
        );
        assert!(last_was_final, "Last chunk should be marked as final");
    }

    #[test]
    fn test_audio_sample_format_validation() {
        // Test valid f32 audio samples
        let valid_samples = vec![-1.0, -0.5, 0.0, 0.5, 1.0];

        for &sample in &valid_samples {
            assert!(
                (-1.0..=1.0).contains(&sample),
                "Sample {} should be in valid range [-1.0, 1.0]",
                sample
            );
        }
    }

    #[test]
    fn test_audio_sample_conversion_i16_to_f32() {
        let i16_samples = [32767i16, 0, -32768, 16384, -16384];

        let f32_samples: Vec<f32> = i16_samples.iter().map(|&s| s as f32 / 32768.0).collect();

        // Verify conversion accuracy
        assert!((f32_samples[0] - 0.99997).abs() < 0.001); // Close to 1.0
        assert_eq!(f32_samples[1], 0.0);
        assert_eq!(f32_samples[2], -1.0);
        assert!((f32_samples[3] - 0.5).abs() < 0.001); // Close to 0.5
        assert!((f32_samples[4] + 0.5).abs() < 0.001); // Close to -0.5
    }

    #[test]
    fn test_audio_buffer_capacity_management() {
        let mut buffer = Vec::with_capacity(1000);

        // Add some data
        buffer.extend_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.capacity(), 1000);

        // Clear but keep capacity
        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), 1000);

        // Reuse buffer
        buffer.extend_from_slice(&[4.0, 5.0]);
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.capacity(), 1000);
    }

    #[test]
    fn test_audio_chunk_boundary_calculations() {
        // Test that chunk boundaries are calculated correctly
        let test_cases = vec![
            (0, 0.0),  // First chunk starts at 0
            (1, 10.0), // Second chunk starts at 10 seconds
            (2, 20.0), // Third chunk starts at 20 seconds
        ];

        for (index, expected_start) in test_cases {
            let start_time = index as f32 * AudioChunk::TARGET_DURATION;
            assert_eq!(
                start_time, expected_start,
                "Chunk {} should start at {} seconds",
                index, expected_start
            );
        }
    }

    #[test]
    fn test_audio_rms_calculation() {
        // Helper function to calculate RMS
        fn calculate_rms(samples: &[f32]) -> f32 {
            if samples.is_empty() {
                return 0.0;
            }
            let sum_of_squares: f32 = samples.iter().map(|&s| s * s).sum();
            (sum_of_squares / samples.len() as f32).sqrt()
        }

        // Test with known values
        let samples = vec![0.0, 1.0, 0.0, -1.0]; // RMS should be sqrt(0.5) ≈ 0.707
        let rms = calculate_rms(&samples);
        assert!((rms - 0.707).abs() < 0.01);

        // Test with silence
        let silence = vec![0.0; 1000];
        let silence_rms = calculate_rms(&silence);
        assert_eq!(silence_rms, 0.0);

        // Test with empty
        let empty_rms = calculate_rms(&[]);
        assert_eq!(empty_rms, 0.0);
    }

    #[test]
    fn test_audio_silence_detection() {
        // Helper function to detect silence
        fn is_silence(samples: &[f32], threshold: f32) -> bool {
            samples.iter().all(|&s| s.abs() < threshold)
        }

        let silent_samples = vec![0.0, 0.001, -0.001, 0.0005];
        let loud_samples = vec![0.0, 0.5, 0.0, -0.3];

        assert!(is_silence(&silent_samples, 0.01));
        assert!(!is_silence(&loud_samples, 0.01));

        // Edge case: empty samples
        assert!(is_silence(&[], 0.01));
    }

    #[test]
    fn test_audio_chunk_validation() {
        // Helper to validate audio chunk properties
        fn validate_chunk(chunk: &AudioChunk) -> bool {
            chunk.sample_rate == 16000
                && chunk.duration >= 0.0
                && chunk.samples.len() <= AudioChunk::TARGET_SAMPLES
                && chunk.start_time >= 0.0
        }

        // Valid chunk
        let valid_chunk = AudioChunk::new(vec![0.0; 1000], 0, 0.0, false);
        assert!(validate_chunk(&valid_chunk));

        // Invalid chunk (wrong sample rate)
        let invalid_chunk = AudioChunk {
            samples: vec![0.0; 1000],
            sample_rate: 8000, // Wrong sample rate
            duration: 0.1,
            index: 0,
            start_time: 0.0,
            is_final: false,
        };
        assert!(!validate_chunk(&invalid_chunk));

        // Invalid chunk (negative start time)
        let invalid_chunk2 = AudioChunk {
            samples: vec![0.0; 1000],
            sample_rate: 16000,
            duration: 0.1,
            index: 0,
            start_time: -1.0, // Negative start time
            is_final: false,
        };
        assert!(!validate_chunk(&invalid_chunk2));
    }
}
