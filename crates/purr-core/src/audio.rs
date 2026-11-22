//! Audio processing functionality using `FFmpeg`

use crate::error::{AudioProcessingError, Error, Result};
use crate::input::{AsyncCustomInput, AsyncStreamBuffer};
use crate::resample::FrameResampler;
use crate::url::{HttpStreamer, UrlStreamConfig};
use ffmpeg_next as ffmpeg;
use futures::Stream;
use reqwest::IntoUrl;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::{self, JoinHandle};
use tracing::{debug, warn};
use url::Url;

const DECODED_CHANNEL_CAPACITY: usize = 8;

const SAMPLE_RATE: u64 = 16_000;

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

/// How much audio a source holds, as far as it says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceLength {
    /// The container states its duration
    Exact(Duration),
    /// Worked out from the size of the body and the bit rate
    Estimated(Duration),
    Unknown,
}

/// Stream of audio chunks
pub struct AudioStream {
    receiver: mpsc::Receiver<Result<AudioChunk>>,
    length: SourceLength,
}

impl AudioStream {
    pub fn new(receiver: mpsc::Receiver<Result<AudioChunk>>, length: SourceLength) -> Self {
        Self { receiver, length }
    }

    #[must_use]
    pub fn length(&self) -> SourceLength {
        self.length
    }

    /// A stream that already holds all of `samples` (mono, 16 kHz), in chunks of the usual size.
    #[must_use]
    pub fn from_samples(samples: &[f32]) -> Self {
        let parts = samples.chunks(AudioChunk::TARGET_SAMPLES).count();
        let (tx, rx) = mpsc::channel(parts.max(1));
        for (index, part) in samples.chunks(AudioChunk::TARGET_SAMPLES).enumerate() {
            let chunk = AudioChunk::new(
                part.to_vec(),
                index,
                index as f32 * AudioChunk::TARGET_DURATION,
                index + 1 == parts,
            );
            if tx.try_send(Ok(chunk)).is_err() {
                break;
            }
        }
        let length = SourceLength::Exact(Duration::from_micros(
            samples.len() as u64 * 1_000_000 / SAMPLE_RATE,
        ));
        Self::new(rx, length)
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
        f.debug_struct("AudioStream")
            .field("length", &self.length)
            .finish_non_exhaustive()
    }
}

fn container_duration(input: &ffmpeg::format::context::Input) -> Option<Duration> {
    u64::try_from(input.duration())
        .ok()
        .filter(|micros| *micros > 0)
        .map(Duration::from_micros)
}

fn estimated_duration(
    input: &ffmpeg::format::context::Input,
    content_length: u64,
) -> Option<Duration> {
    let stream_rate = input
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .map(|stream| stream.parameters().bit_rate());
    let bit_rate = [Some(input.bit_rate()), stream_rate]
        .into_iter()
        .flatten()
        .find(|rate| *rate > 0)?;
    Duration::try_from_secs_f64(content_length as f64 * 8.0 / bit_rate as f64).ok()
}

fn join_error(source: task::JoinError) -> Error {
    Error::from(AudioProcessingError::TaskJoin { source })
}

/// Audio processor using `FFmpeg`
pub struct AudioProcessor {}

impl AudioProcessor {
    /// Create a new audio processor
    pub fn new() -> Result<Self> {
        ffmpeg::init().map_err(|source| Error::InitFfmpeg { source })?;

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
            let mut processor = Self::new()?;
            processor.load_audio_sync(&path)
        })
        .await
        .map_err(join_error)?
    }

    /// How long the audio file at `path` is, as its container states it.
    pub fn probe_file_length(path: &Path) -> Result<SourceLength> {
        Self::new()?;
        let (_input, length) = Self::open_file(path)?;
        Ok(length)
    }

    /// Stream audio file as chunks for real-time processing
    ///
    /// The file is opened before this returns, so a file that cannot be read fails here.
    pub async fn stream_file<P: AsRef<Path>>(path: P) -> Result<AudioStream> {
        let path = path.as_ref().to_path_buf();

        let (mut processor, mut input, length) = task::spawn_blocking(move || {
            let processor = Self::new()?;
            let (input, length) = Self::open_file(&path)?;
            Ok::<_, Error>((processor, input, length))
        })
        .await
        .map_err(join_error)??;

        let (tx, rx) = mpsc::channel(DECODED_CHANNEL_CAPACITY);

        task::spawn_blocking(move || {
            if let Err(error) = processor.stream_audio_from_input(&mut input, &tx, &|| Ok(())) {
                Self::send_failure(&tx, error);
            }
        });

        Ok(AudioStream::new(rx, length))
    }

    /// Ends the stream with `error`, so whoever reads the stream sees why it stopped.
    fn send_failure(tx: &mpsc::Sender<Result<AudioChunk>>, error: Error) {
        if tx.blocking_send(Err(error)).is_err() {
            debug!("Audio stream dropped, failure not delivered");
        }
    }

    fn open_file(path: &Path) -> Result<(ffmpeg::format::context::Input, SourceLength)> {
        if !path.exists() {
            return Err(Error::from(AudioProcessingError::ReadFailed {
                file: path.to_path_buf(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "File not found"),
            }));
        }

        let input = ffmpeg::format::input(&path).map_err(|source| Error::OpenAudioInput {
            path: path.to_path_buf(),
            source,
        })?;
        let length = container_duration(&input).map_or(SourceLength::Unknown, SourceLength::Exact);
        Ok((input, length))
    }

    /// Synchronous audio loading implementation
    fn load_audio_sync(&mut self, path: &Path) -> Result<AudioData> {
        let (mut input, _length) = Self::open_file(path)?;
        self.load_audio_from_input(&mut input, &|| Ok(()))
    }

    fn url_stream_config(config: &crate::config::InputConfig) -> UrlStreamConfig {
        UrlStreamConfig {
            connect_timeout: Duration::from_secs(config.http_connect_timeout),
            read_timeout: Duration::from_secs(config.http_timeout),
            ..Default::default()
        }
    }

    /// Fails with the reason the download broke off, if it did.
    fn source_failure(buffer: &AsyncStreamBuffer) -> Result<()> {
        buffer
            .failure()
            .map_or_else(|| Ok(()), |message| Err(Error::NetworkError { message }))
    }

    async fn open_stream(buffer: &Arc<AsyncStreamBuffer>) -> Result<AsyncCustomInput> {
        match AsyncCustomInput::create(Arc::clone(buffer)).await {
            Ok(input) => Ok(input),
            Err(source) => {
                Self::source_failure(buffer)?;
                Err(Error::OpenStreamInput { source })
            }
        }
    }

    fn watch_download(task: JoinHandle<Result<()>>, buffer: Arc<AsyncStreamBuffer>) {
        tokio::spawn(async move {
            match task.await {
                Ok(Ok(())) => debug!("The download finished"),
                Ok(Err(error)) => debug!("The download ended with an error: {error}"),
                Err(source) => {
                    warn!("The download task did not finish: {source}");
                    buffer.fail(format!("The download task did not finish: {source}"));
                }
            }
        });
    }

    /// Load audio from URL and convert to the format expected by Whisper
    pub async fn load_audio_from_url(url: impl IntoUrl) -> Result<AudioData> {
        Self::load_audio_from_url_with_config(
            url.into_url()
                .map_err(|source| crate::error::Error::InvalidUrl { source })?,
            &crate::config::InputConfig::default(),
        )
        .await
    }

    /// Load audio from URL with custom configuration
    pub async fn load_audio_from_url_with_config(
        url: Url,
        config: &crate::config::InputConfig,
    ) -> Result<AudioData> {
        let streamer = HttpStreamer::with_config(Self::url_stream_config(config))?;
        let buffer = AsyncStreamBuffer::new();
        let download = streamer.stream_url(url, Arc::clone(&buffer)).await?;
        Self::watch_download(download.task, Arc::clone(&buffer));

        let mut custom_input = Self::open_stream(&buffer).await?;

        task::spawn_blocking(move || {
            let mut processor = Self::new()?;
            processor
                .load_audio_from_input(custom_input.input_mut(), &|| Self::source_failure(&buffer))
        })
        .await
        .map_err(join_error)?
    }

    /// Stream audio from URL as chunks for real-time processing
    pub async fn stream_url(url: Url) -> Result<AudioStream> {
        Self::stream_url_with_config(url, &crate::config::InputConfig::default()).await
    }

    /// Stream audio from URL with custom configuration
    ///
    /// The stream is opened and probed before this returns.
    pub async fn stream_url_with_config(
        url: Url,
        config: &crate::config::InputConfig,
    ) -> Result<AudioStream> {
        let streamer = HttpStreamer::with_config(Self::url_stream_config(config))?;
        let buffer = AsyncStreamBuffer::new();
        let download = streamer.stream_url(url, Arc::clone(&buffer)).await?;
        let content_length = download.content_length;
        Self::watch_download(download.task, Arc::clone(&buffer));

        let mut custom_input = Self::open_stream(&buffer).await?;
        let length = container_duration(custom_input.input()).map_or_else(
            || {
                content_length
                    .and_then(|bytes| estimated_duration(custom_input.input(), bytes))
                    .map_or(SourceLength::Unknown, SourceLength::Estimated)
            },
            SourceLength::Exact,
        );

        let (tx, rx) = mpsc::channel(DECODED_CHANNEL_CAPACITY);

        task::spawn_blocking(move || {
            let mut processor = match Self::new() {
                Err(e) => {
                    Self::send_failure(&tx, e);
                    return;
                }
                Ok(p) => p,
            };

            if let Err(error) =
                processor.stream_audio_from_input(custom_input.input_mut(), &tx, &|| {
                    Self::source_failure(&buffer)
                })
            {
                Self::send_failure(&tx, error);
            }
        });

        Ok(AudioStream::new(rx, length))
    }

    /// Load audio from an opened input
    ///
    /// `check_source` runs once the input has ended and fails if it ended early.
    fn load_audio_from_input(
        &mut self,
        ictx: &mut ffmpeg::format::context::Input,
        check_source: &dyn Fn() -> Result<()>,
    ) -> Result<AudioData> {
        // Find the audio stream
        let input_stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .ok_or_else(|| Error::from(AudioProcessingError::NoAudioStream))?;

        let stream_index = input_stream.index();

        // Get decoder
        let context_decoder =
            ffmpeg::codec::context::Context::from_parameters(input_stream.parameters())
                .map_err(|source| Error::CreateAudioDecoder { source })?;

        let mut decoder = context_decoder
            .decoder()
            .audio()
            .map_err(|source| Error::CreateAudioDecoder { source })?;

        let mut samples = Vec::new();
        let mut frame = ffmpeg::frame::Audio::empty();
        let mut resampler = FrameResampler::new();

        // Process packets with error resilience
        for (stream, packet) in ictx.packets() {
            if stream.index() == stream_index {
                match decoder.send_packet(&packet) {
                    Ok(()) => {
                        while decoder.receive_frame(&mut frame).is_ok() {
                            resampler.push(&frame, &mut samples)?;
                        }
                    }
                    Err(ffmpeg_next::Error::InvalidData) => {
                        warn!("Skipping invalid chunk at stream index {}", stream_index);
                        continue;
                    }
                    Err(source) => {
                        return Err(Error::DecodeAudioPacket {
                            stream_index,
                            source,
                        });
                    }
                }
            }
        }

        check_source()?;

        // Flush decoder
        match decoder.send_eof() {
            Ok(()) => {
                while decoder.receive_frame(&mut frame).is_ok() {
                    resampler.push(&frame, &mut samples)?;
                }
            }
            Err(e) => {
                warn!("Failed to flush the decoder, but continuing: {e}");
            }
        }

        resampler.finish(&mut samples)?;

        // Check if we got any audio data
        if samples.is_empty() {
            return Err(Error::from(AudioProcessingError::NoAudioData));
        }

        let duration = samples.len() as f32 / 16000.0;

        Ok(AudioData {
            samples,
            sample_rate: 16000,
            duration,
        })
    }

    /// Send every complete chunk buffered in `output_samples`; false once the receiver is gone.
    fn send_full_chunks(
        output_samples: &mut Vec<f32>,
        chunk_index: &mut usize,
        total_processed_time: &mut f32,
        tx: &mpsc::Sender<Result<AudioChunk>>,
    ) -> bool {
        while output_samples.len() >= AudioChunk::TARGET_SAMPLES {
            let chunk_samples: Vec<f32> =
                output_samples.drain(..AudioChunk::TARGET_SAMPLES).collect();
            let chunk = AudioChunk::new(chunk_samples, *chunk_index, *total_processed_time, false);
            *total_processed_time += chunk.duration;
            *chunk_index += 1;

            if tx.blocking_send(Ok(chunk)).is_err() {
                debug!("Audio stream dropped, stopping the decode");
                return false;
            }
        }
        true
    }

    /// Stream audio from an opened input for real-time processing
    ///
    /// Waits while the receiver holds as many chunks as the channel takes. `check_source` runs
    /// once the input has ended, before the last chunk goes out, and fails if it ended early.
    fn stream_audio_from_input(
        &mut self,
        ictx: &mut ffmpeg::format::context::Input,
        tx: &mpsc::Sender<Result<AudioChunk>>,
        check_source: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        // Find the audio stream
        let input_stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .ok_or_else(|| Error::from(AudioProcessingError::NoAudioStream))?;

        let stream_index = input_stream.index();

        // Get decoder
        let context_decoder =
            ffmpeg::codec::context::Context::from_parameters(input_stream.parameters())
                .map_err(|source| Error::CreateAudioDecoder { source })?;

        let mut decoder = context_decoder
            .decoder()
            .audio()
            .map_err(|source| Error::CreateAudioDecoder { source })?;

        let mut output_samples = Vec::new();
        let mut frame = ffmpeg::frame::Audio::empty();
        let mut resampler = FrameResampler::new();

        let mut chunk_index = 0;
        let mut total_processed_time = 0.0;

        // Process packets with error resilience
        for (stream, packet) in ictx.packets() {
            if stream.index() == stream_index {
                match decoder.send_packet(&packet) {
                    Ok(()) => {
                        while decoder.receive_frame(&mut frame).is_ok() {
                            resampler.push(&frame, &mut output_samples)?;

                            if !Self::send_full_chunks(
                                &mut output_samples,
                                &mut chunk_index,
                                &mut total_processed_time,
                                tx,
                            ) {
                                return Ok(());
                            }
                        }
                    }
                    Err(ffmpeg_next::Error::InvalidData) => {
                        warn!("Skipping invalid chunk at stream index {}", stream_index);
                        continue;
                    }
                    Err(source) => {
                        return Err(Error::DecodeAudioPacket {
                            stream_index,
                            source,
                        });
                    }
                }
            }
        }

        check_source()?;

        // Flush decoder
        match decoder.send_eof() {
            Ok(()) => {
                while decoder.receive_frame(&mut frame).is_ok() {
                    resampler.push(&frame, &mut output_samples)?;

                    if !Self::send_full_chunks(
                        &mut output_samples,
                        &mut chunk_index,
                        &mut total_processed_time,
                        tx,
                    ) {
                        return Ok(());
                    }
                }
            }
            Err(e) => {
                warn!("Failed to flush the decoder, but continuing: {e}");
            }
        }

        resampler.finish(&mut output_samples)?;
        if !Self::send_full_chunks(
            &mut output_samples,
            &mut chunk_index,
            &mut total_processed_time,
            tx,
        ) {
            return Ok(());
        }

        // Send remaining samples as final chunk
        if !output_samples.is_empty() {
            let final_chunk =
                AudioChunk::new(output_samples, chunk_index, total_processed_time, true);
            if tx.blocking_send(Ok(final_chunk)).is_err() {
                debug!("Audio stream dropped, final chunk not delivered");
            }
        } else if chunk_index == 0 {
            return Err(Error::from(AudioProcessingError::NoAudioData));
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

        let (tx, rx) = mpsc::channel(1);
        let stream = AudioStream::new(rx, SourceLength::Unknown);

        // Create a simple test by dropping the stream immediately
        drop(stream);
        drop(tx);
    }

    #[tokio::test]
    async fn test_audio_stream_receiving() {
        use futures::StreamExt;
        use tokio::sync::mpsc;

        let (tx, rx) = mpsc::channel(1);
        let mut stream = AudioStream::new(rx, SourceLength::Unknown);

        // Send a test chunk
        let test_chunk = AudioChunk::new(vec![0.1, 0.2, 0.3], 0, 0.0, true);
        tx.send(Ok(test_chunk.clone())).await.unwrap();
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

        let (tx, rx) = mpsc::channel(1);
        let mut stream = AudioStream::new(rx, SourceLength::Unknown);

        // Send an error
        let test_error = Error::from(AudioProcessingError::processing_failed(
            "test operation",
            std::io::Error::other("Test error"),
        ));
        tx.send(Err(test_error)).await.unwrap();
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
            Error::AudioProcessing {
                source: AudioProcessingError::ReadFailed { .. },
            } => {
                // Expected ReadFailed error for nonexistent file
            }
            Error::AudioProcessing {
                source: AudioProcessingError::ProcessingFailed { operation, .. },
            } => {
                assert!(operation.contains("not found"));
            }
            _ => panic!(
                "Expected ReadFailed or ProcessingFailed error with 'not found', got: {error:?}"
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
        assert_ne!(audio_data.samples, [] as [f32; 0]);
    }

    #[tokio::test]
    async fn test_stream_nonexistent_file() {
        use futures::StreamExt;

        let result = AudioProcessor::stream_file("nonexistent_file.wav").await;

        // Stream creation might succeed, but the first chunk should contain an error
        if let Ok(mut stream) = result {
            // Try to get the first chunk - this should fail
            if let Some(chunk_result) = stream.next().await {
                assert!(
                    chunk_result.is_err(),
                    "Should get an error for nonexistent file"
                );
            }
        } else {
            // Also acceptable - immediate failure
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
                    assert_ne!(chunk.samples, [] as [f32; 0]);

                    last_was_final = chunk.is_final;
                    chunk_count += 1;

                    if chunk.is_final {
                        break;
                    }
                }
                Err(e) => panic!("Stream error: {e:?}"),
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
                "Sample {sample} should be in valid range [-1.0, 1.0]"
            );
        }
    }

    #[test]
    fn test_audio_sample_conversion_i16_to_f32() {
        let i16_samples = [32767i16, 0, -32768, 16384, -16384];

        let f32_samples: Vec<f32> = i16_samples
            .iter()
            .map(|&s| f32::from(s) / 32768.0)
            .collect();

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
                "Chunk {index} should start at {expected_start} seconds"
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
