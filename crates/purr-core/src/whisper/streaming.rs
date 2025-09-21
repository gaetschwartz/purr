use crate::{
    error::TranscriptionError,
    whisper::{
        load_model, StreamingChunk, TranscriptionResult, TranscriptionStats, WhisperTranscriber,
    },
    AudioStream, ModelManager, TranscriptionConfig,
};
use futures::{Stream, StreamExt};
use std::{
    pin::Pin,
    task::{Context, Poll},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::warn;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

pub struct StreamWhisperTranscriber {
    context: WhisperContext,
    config: TranscriptionConfig,
}

impl WhisperTranscriber for StreamWhisperTranscriber {
    type TranscriberResult = StreamingTranscriptionResult;
    type InputData = AudioStream;

    async fn from_config(config: crate::TranscriptionConfig) -> crate::Result<Self>
    where
        Self: Sized,
    {
        let config_clone = config.clone();

        let model_manager = ModelManager::new()?;

        // Load the model (which may involve async model discovery)
        let context = load_model(&config_clone, &model_manager).await?;

        Ok(Self { context, config })
    }

    async fn transcribe(self, input: AudioStream) -> crate::Result<StreamingTranscriptionResult> {
        let (tx, rx) = mpsc::unbounded_channel();

        // Spawn background task to process audio stream
        tokio::spawn(async move {
            let mut transcriber = self;
            if let Err(e) = transcriber.process_audio_stream(input, tx.clone()).await {
                let _ = tx.send(Err(e));
            }
        });

        Ok(StreamingTranscriptionResult {
            stream: Box::pin(UnboundedReceiverStream::new(rx)),
        })
    }
}

impl StreamWhisperTranscriber {
    async fn process_audio_stream(
        &mut self,
        mut input: AudioStream,
        tx: mpsc::UnboundedSender<crate::Result<StreamingChunk>>,
    ) -> crate::Result<()> {
        // Statistics tracking
        let start_time = std::time::Instant::now();
        let mut total_audio_duration = 0.0f32;
        let mut accumulated_samples = Vec::new();

        // Collect all audio chunks first
        while let Some(chunk_result) = input.next().await {
            match chunk_result {
                Ok(audio_chunk) => {
                    total_audio_duration += audio_chunk.duration;
                    accumulated_samples.extend_from_slice(&audio_chunk.samples);

                    // If this is the final chunk, process all accumulated audio
                    if audio_chunk.is_final {
                        // Create a state for processing complete audio
                        let mut state = self.context.create_state().map_err(|e| {
                            crate::WhisperError::from(TranscriptionError::StateCreation { source: e })
                        })?;

                        // Create params with proper whisper.cpp defaults
                        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

                        // Configure parameters to match whisper.cpp behavior
                        params.set_language(self.config.language.as_deref());
                        params.set_translate(self.config.translate);

                        if let Some(threads) = self.config.num_threads {
                            params.set_n_threads(threads as i32);
                        }

                        params.set_temperature(self.config.temperature);
                        params.set_print_timestamps(false);
                        params.set_print_progress(false);
                        params.set_print_special(false);
                        params.set_print_realtime(false);

                        // Process the complete audio data (like non-streaming mode)
                        match state.full(params, &accumulated_samples) {
                            Ok(_) => {
                                // Extract results from state
                                let num_segments = state.full_n_segments();
                                let mut full_text = String::new();

                                for i in 0..num_segments {
                                    if let Some(segment) = state.get_segment(i) {
                                        match segment.to_str() {
                                            Ok(text) => full_text.push_str(text),
                                            Err(e) => {
                                                warn!("Failed to get segment text: {}", e);
                                            }
                                        }
                                    } else {
                                        warn!("Failed to get segment {} (out of {})", i, num_segments);
                                    }
                                }

                                // Calculate final statistics
                                let processing_time = start_time.elapsed().as_secs_f64();
                                let word_count = full_text.split_whitespace().count();
                                let final_stats = Some(TranscriptionStats {
                                    processing_time,
                                    audio_duration: total_audio_duration,
                                    segment_count: num_segments as usize,
                                    word_count,
                                });

                                // Send the final result as a single chunk (maintaining streaming interface)
                                let streaming_chunk = StreamingChunk {
                                    text: full_text,
                                    start: 0.0,
                                    end: f64::from(total_audio_duration),
                                    is_final: true,
                                    chunk_index: 0,
                                    final_stats,
                                };

                                if tx.send(Ok(streaming_chunk)).is_err() {
                                    // Receiver dropped, stop processing
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Transcription failed: {}", e);
                                let _ = tx.send(Err(crate::WhisperError::from(TranscriptionError::Failed { source: e })));
                            }
                        }
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                    break;
                }
            }
        }

        Ok(())
    }
}

pub struct StreamingTranscriptionResult {
    stream: Pin<Box<dyn Stream<Item = Result<StreamingChunk, crate::WhisperError>> + Send>>,
}

impl Stream for StreamingTranscriptionResult {
    type Item = Result<StreamingChunk, crate::WhisperError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}

impl TranscriptionResult for StreamingTranscriptionResult {}
