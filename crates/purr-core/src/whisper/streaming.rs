use crate::{
    error::TranscriptionError,
    whisper::{
        load_model, StreamingChunk, TranscriptionResult, TranscriptionStats, WhisperTranscriber,
    },
    AudioStream, ModelManager, TranscriptionConfig,
};
use futures::{Stream, StreamExt};
use std::{
    collections::VecDeque,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, warn};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperState};

pub struct StreamWhisperTranscriber {
    context: WhisperContext,
    config: Arc<TranscriptionConfig>,
}

/// Streaming state for managing audio buffers and context
struct StreamingState {
    /// Main audio buffer for processing
    audio_buffer: VecDeque<f32>,
    /// Previous audio chunk for context continuity (overlapping window)
    context_buffer: Vec<f32>,
    /// Whisper processing state
    whisper_state: WhisperState,
    /// Current chunk index
    chunk_index: usize,
    /// Total processed audio duration
    total_duration: f32,
    /// Start time for current processing window
    window_start_time: f32,
    /// Buffer size for processing chunks (in samples)
    chunk_size: usize,
    /// Context size to retain between chunks (in samples)
    context_size: usize,
}

impl WhisperTranscriber for StreamWhisperTranscriber {
    type TranscriberResult = StreamingTranscription;
    type InputData = AudioStream;

    async fn from_config(config: crate::TranscriptionConfig) -> crate::Result<Self>
    where
        Self: Sized,
    {
        let config = Arc::new(config);
        let model_manager = ModelManager::new()?;

        // Load the model (which may involve async model discovery)
        let context = load_model(&config, &model_manager).await?;

        Ok(Self { context, config })
    }

    async fn transcribe(self, input: AudioStream) -> crate::Result<StreamingTranscription> {
        let (tx, rx) = mpsc::unbounded_channel();

        // Spawn background task to process audio stream
        tokio::spawn(async move {
            let mut transcriber = self;
            if let Err(e) = transcriber.process_audio_stream(input, tx.clone()).await {
                let _ = tx.send(Err(e));
            }
        });

        Ok(StreamingTranscription {
            stream: Box::pin(UnboundedReceiverStream::new(rx)),
        })
    }
}

impl StreamingState {
    fn new(context: &WhisperContext, config: Arc<TranscriptionConfig>) -> crate::Result<Self> {
        let whisper_state = context.create_state().map_err(|e| {
            crate::WhisperError::from(TranscriptionError::StateCreation { source: e })
        })?;

        // Stream processing configuration - optimized for performance vs quality balance
        // 4-second chunks with minimal overlap for better performance
        let chunk_size = (config.sample_rate as f32 * config.chunk_size) as usize; // 4 seconds at 16kHz - good for beam search quality
        let context_size = (config.sample_rate as f32 * config.chunk_overlap) as usize; // 0.5 seconds overlap for context continuity

        Ok(Self {
            audio_buffer: VecDeque::new(),
            context_buffer: Vec::new(),
            whisper_state,
            chunk_index: 0,
            total_duration: 0.0,
            window_start_time: 0.0,
            chunk_size,
            context_size,
        })
    }

    fn add_audio(&mut self, samples: &[f32]) {
        self.audio_buffer.extend(samples);
    }

    fn should_process(&self) -> bool {
        self.audio_buffer.len() >= self.chunk_size
    }

    fn prepare_processing_buffer(&mut self) -> Vec<f32> {
        // Create processing buffer with context + new audio
        let mut processing_buffer = Vec::with_capacity(self.context_buffer.len() + self.chunk_size);

        // Add previous context for continuity
        processing_buffer.extend_from_slice(&self.context_buffer);

        // Add new audio chunk
        let chunk_samples: Vec<f32> = self
            .audio_buffer
            .drain(..self.chunk_size.min(self.audio_buffer.len()))
            .collect();

        processing_buffer.extend_from_slice(&chunk_samples);

        // Update context buffer with the end of current chunk for next iteration
        let context_start = chunk_samples.len().saturating_sub(self.context_size);
        self.context_buffer = chunk_samples[context_start..].to_vec();

        processing_buffer
    }

    fn update_timing(&mut self, processed_samples: usize) {
        let processed_duration = processed_samples as f32 / 16000.0;
        self.total_duration += processed_duration;
        self.window_start_time = self.total_duration - (self.context_buffer.len() as f32 / 16000.0);
        self.chunk_index += 1;
    }
}

impl StreamWhisperTranscriber {
    async fn process_audio_stream(
        &mut self,
        mut input: AudioStream,
        tx: mpsc::UnboundedSender<crate::Result<StreamingChunk>>,
    ) -> crate::Result<()> {
        let processing_start = std::time::Instant::now();
        let mut streaming_state = StreamingState::new(&self.context, self.config.clone())?;
        let mut is_stream_complete = false;

        debug!("Starting progressive audio stream processing");

        // Process audio chunks as they arrive
        while let Some(chunk_result) = input.next().await {
            match chunk_result {
                Ok(audio_chunk) => {
                    debug!(
                        "Received audio chunk {} with {} samples, duration: {:.2}s, is_final: {}",
                        audio_chunk.index,
                        audio_chunk.samples.len(),
                        audio_chunk.duration,
                        audio_chunk.is_final
                    );

                    // Add audio samples to the streaming buffer
                    streaming_state.add_audio(&audio_chunk.samples);

                    // Mark if this is the final chunk
                    if audio_chunk.is_final {
                        is_stream_complete = true;
                    }

                    // Process accumulated audio when we have enough samples
                    while streaming_state.should_process()
                        || (is_stream_complete && !streaming_state.audio_buffer.is_empty())
                    {
                        let processing_buffer = streaming_state.prepare_processing_buffer();

                        if processing_buffer.is_empty() {
                            break;
                        }

                        let buffer_len = processing_buffer.len();
                        let is_final_chunk =
                            is_stream_complete && streaming_state.audio_buffer.is_empty();

                        // Process the audio chunk through whisper
                        match self
                            .process_audio_chunk(
                                &mut streaming_state,
                                processing_buffer,
                                is_final_chunk,
                            )
                            .await
                        {
                            Ok(Some(chunk)) => {
                                debug!(
                                    "Emitting streaming chunk {} with text: '{}'",
                                    chunk.chunk_index,
                                    chunk.text.trim()
                                );
                                if tx.send(Ok(chunk)).is_err() {
                                    debug!("Receiver dropped, stopping stream processing");
                                    return Ok(());
                                }
                            }
                            Ok(None) => {
                                // No transcription result for this chunk (silent audio, etc.)
                                debug!(
                                    "No transcription result for chunk {}",
                                    streaming_state.chunk_index
                                );
                            }
                            Err(e) => {
                                warn!("Failed to process audio chunk: {}", e);
                                let _ = tx.send(Err(e));
                                return Ok(());
                            }
                        }

                        streaming_state.update_timing(buffer_len);
                    }

                    // If this was the final chunk and we've processed everything, break
                    if is_stream_complete && streaming_state.audio_buffer.is_empty() {
                        break;
                    }
                }
                Err(e) => {
                    warn!("Audio stream error: {}", e);
                    let _ = tx.send(Err(e));
                    break;
                }
            }
        }

        // Send final statistics if we completed successfully
        if is_stream_complete {
            let final_processing_time = processing_start.elapsed().as_secs_f64();
            debug!(
                "Stream processing completed. Total duration: {:.2}s, processing time: {:.2}s",
                streaming_state.total_duration, final_processing_time
            );
        }

        Ok(())
    }

    async fn process_audio_chunk(
        &mut self,
        streaming_state: &mut StreamingState,
        audio_buffer: Vec<f32>,
        is_final_chunk: bool,
    ) -> crate::Result<Option<StreamingChunk>> {
        if audio_buffer.is_empty() {
            return Ok(None);
        }

        // Create whisper processing parameters with proper sampling strategy
        // Match whisper.cpp's "4 threads, 1 processors, 5 beams + best of 5" quality
        let beam_size = self.config.beam_size.unwrap_or(5) as i32;
        let mut params = if beam_size > 1 {
            FullParams::new(SamplingStrategy::BeamSearch {
                beam_size,
                patience: -1.0,
            })
        } else {
            FullParams::new(SamplingStrategy::Greedy { best_of: 5 })
        };

        // Configure parameters for streaming (based on whisper.cpp)
        params.set_language(self.config.language.as_deref());
        params.set_translate(self.config.translate);
        params.set_temperature(self.config.temperature);

        if let Some(threads) = self.config.num_threads {
            params.set_n_threads(threads as i32);
        }

        // Streaming-specific settings
        params.set_print_timestamps(false);
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_no_context(false); // Keep context for continuity - CRITICAL for full transcription
        params.set_single_segment(false); // Allow multiple segments

        // Process audio through whisper
        match streaming_state.whisper_state.full(params, &audio_buffer) {
            Ok(_) => {
                let num_segments = streaming_state.whisper_state.full_n_segments();

                if num_segments == 0 {
                    // No speech detected in this chunk
                    return Ok(None);
                }

                // Extract transcription text from all segments
                let mut chunk_text = String::new();
                let mut chunk_start_time = f64::MAX;
                let mut chunk_end_time: f64 = 0.0;

                for i in 0..num_segments {
                    if let Some(segment) = streaming_state.whisper_state.get_segment(i) {
                        // Get segment timing (whisper provides relative times)
                        let segment_start = segment.start_timestamp() as f64 / 100.0; // Convert from centiseconds
                        let segment_end = segment.end_timestamp() as f64 / 100.0;

                        // Adjust timing based on our streaming window
                        let adjusted_start =
                            f64::from(streaming_state.window_start_time) + segment_start;
                        let adjusted_end =
                            f64::from(streaming_state.window_start_time) + segment_end;

                        chunk_start_time = chunk_start_time.min(adjusted_start);
                        chunk_end_time = chunk_end_time.max(adjusted_end);

                        match segment.to_str() {
                            Ok(text) => {
                                let trimmed_text = text.trim();
                                if !trimmed_text.is_empty() {
                                    if !chunk_text.is_empty() {
                                        chunk_text.push(' ');
                                    }
                                    chunk_text.push_str(trimmed_text);
                                }
                            }
                            Err(e) => {
                                warn!("Failed to extract segment text: {}", e);
                            }
                        }
                    }
                }

                // Only return a chunk if we have actual text
                if chunk_text.trim().is_empty() {
                    return Ok(None);
                }

                // Prepare final stats for the last chunk
                let final_stats = if is_final_chunk {
                    let total_processing_time = std::time::Instant::now()
                        .duration_since(
                            std::time::Instant::now()
                                .checked_sub(std::time::Duration::from_secs_f32(
                                    streaming_state.total_duration,
                                ))
                                .unwrap(),
                        )
                        .as_secs_f64();
                    let word_count = chunk_text.split_whitespace().count();

                    Some(TranscriptionStats {
                        processing_time: total_processing_time,
                        audio_duration: streaming_state.total_duration,
                        segment_count: num_segments as usize,
                        word_count,
                    })
                } else {
                    None
                };

                // Ensure we have valid timing
                if chunk_start_time == f64::MAX {
                    chunk_start_time = f64::from(streaming_state.window_start_time);
                }
                if chunk_end_time == 0.0 {
                    chunk_end_time = f64::from(
                        streaming_state.window_start_time + (audio_buffer.len() as f32 / 16000.0),
                    );
                }

                let streaming_chunk = StreamingChunk {
                    text: chunk_text,
                    start: chunk_start_time,
                    end: chunk_end_time,
                    is_final: is_final_chunk,
                    chunk_index: streaming_state.chunk_index,
                    final_stats,
                };

                Ok(Some(streaming_chunk))
            }
            Err(e) => Err(crate::WhisperError::from(TranscriptionError::Failed {
                source: e,
            })),
        }
    }
}

pub struct StreamingTranscription {
    stream: Pin<Box<dyn Stream<Item = Result<StreamingChunk, crate::WhisperError>> + Send>>,
}

impl Stream for StreamingTranscription {
    type Item = Result<StreamingChunk, crate::WhisperError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}

impl TranscriptionResult for StreamingTranscription {}
