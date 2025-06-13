use crate::{
    whisper::{load_model, StreamingChunk, TranscriptionResult, WhisperTranscriber},
    AudioStream, ModelManager, TranscriptionConfig,
};
use futures::{Stream, StreamExt};
use std::{future::Future, pin::Pin, task::{Context, Poll}};
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
        // Install logging hooks to redirect whisper.cpp output to Rust logging
        whisper_rs::install_logging_hooks();

        let config_clone = config.clone();

        let model_manager = ModelManager::new()?;

        // Load the model (which may involve async model discovery)
        let context = load_model(&config_clone, &model_manager).await?;

        Ok(Self { context, config })
    }

    fn transcribe(
        &mut self,
        input: AudioStream,
    ) -> impl Future<Output = crate::Result<StreamingTranscriptionResult>> + '_ {
        async move {
            let (tx, rx) = mpsc::unbounded_channel();
            
            // Process the stream inline since we can't move the context
            let result = self.process_audio_stream(input, tx.clone()).await;
            
            if let Err(e) = result {
                let _ = tx.send(Err(e));
            }

            Ok(StreamingTranscriptionResult {
                stream: Box::pin(UnboundedReceiverStream::new(rx)),
            })
        }
    }
}

impl StreamWhisperTranscriber {
    async fn process_audio_stream(
        &mut self,
        mut input: AudioStream,
        tx: mpsc::UnboundedSender<crate::Result<StreamingChunk>>,
    ) -> crate::Result<()> {
        // Create a state for processing all chunks
        let mut state = self
            .context
            .create_state()
            .map_err(|e| crate::WhisperError::Transcription(format!("Failed to create state: {}", e)))?;

        // Process each audio chunk
        while let Some(chunk_result) = input.next().await {
            match chunk_result {
                Ok(audio_chunk) => {
                    // Create fresh params for each chunk
                    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

                    // Configure parameters
                    if let Some(lang) = &self.config.language {
                        params.set_language(Some(lang));
                    }

                    if let Some(threads) = self.config.num_threads {
                        params.set_n_threads(threads as i32);
                    }

                    params.set_temperature(self.config.temperature);
                    params.set_print_timestamps(false);
                    params.set_print_progress(false);
                    params.set_print_special(false);
                    params.set_print_realtime(false);

                    // Process this chunk
                    match state.full(params, &audio_chunk.samples) {
                        Ok(_) => {
                            // Extract results from state
                            match state.full_n_segments() {
                                Ok(num_segments) => {
                                    let mut chunk_text = String::new();

                                    for i in 0..num_segments {
                                        match state.full_get_segment_text(i) {
                                            Ok(text) => {
                                                chunk_text.push_str(&text);
                                            }
                                            Err(e) => {
                                                warn!("Failed to get segment text for segment {}: {}", i, e);
                                            }
                                        }
                                    }

                                    // Send the chunk result
                                    let streaming_chunk = StreamingChunk {
                                        text: chunk_text,
                                        start: audio_chunk.start_time as f64,
                                        end: (audio_chunk.start_time + audio_chunk.duration) as f64,
                                        is_final: audio_chunk.is_final,
                                        chunk_index: audio_chunk.index,
                                    };

                                    if tx.send(Ok(streaming_chunk)).is_err() {
                                        // Receiver dropped, stop processing
                                        break;
                                    }
                                }
                                Err(e) => {
                                    warn!("Failed to get segment count: {}", e);
                                    // Continue with next chunk
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Transcription failed for chunk {}: {}", audio_chunk.index, e);
                            // Continue with next chunk instead of failing completely
                        }
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

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(cx)
    }
}

impl TranscriptionResult for StreamingTranscriptionResult {}
