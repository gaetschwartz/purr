use dioxus::prelude::*;
use dioxus_fullstack::{codec::JsonEncoding, BoxedStream, Websocket};
use futures::SinkExt as _;
use serde::{Deserialize, Serialize};

#[cfg(feature = "server")]
use {
    purr_core::{transcribe_file_stream, AudioProcessor, TranscriptionConfig},
    std::{env, path::PathBuf, sync::LazyLock},
    tokio,
    tracing::{error, info},
};

/// Transcription request
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TranscriptionRequest {
    /// File ID from upload
    pub file_id: String,
    /// Optional language (auto-detect if None)
    pub language: Option<String>,
    /// Whether to translate to English
    pub translate: bool,
}

/// Transcription status updates
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum TranscriptionStatus {
    /// Starting transcription
    Starting,
    /// Processing audio file
    ProcessingAudio,
    /// Transcription in progress with chunk updates
    InProgress {
        chunk_index: usize,
        text: String,
        start_time: f64,
        end_time: f64,
    },
    /// Final result
    Completed {
        processing_time: f64,
        audio_duration: f32,
        word_count: usize,
    },
    /// Error occurred
    Error { message: String },
}

/// Helper function to get file path from file_id
fn get_file_path(file_id: &str) -> PathBuf {
    static UPLOAD_PATH: LazyLock<PathBuf> = LazyLock::new(|| env::temp_dir().join("purr-uploads"));
    UPLOAD_PATH.join(file_id)
}

/// Start transcription with streaming status updates
#[server(protocol = Websocket<JsonEncoding, JsonEncoding>)]
pub async fn start_transcription(
    input: BoxedStream<TranscriptionRequest, ServerFnError>,
) -> ServerFnResult<BoxedStream<TranscriptionStatus, ServerFnError>> {
    let (tx, rx) = futures::channel::mpsc::channel(10);
    tokio::spawn(async move {
        let mut tx = tx;
        if let Err(e) = audio_transcription_task(input, &mut tx).await {
            error!("Transcription task error: {}", e);
            let _ = tx
                .send(Ok(TranscriptionStatus::Error {
                    message: e.to_string(),
                }))
                .await;
        }
    });

    Ok(rx.into())
}

async fn audio_transcription_task(
    mut input: BoxedStream<TranscriptionRequest, ServerFnError>,
    tx: &mut futures::channel::mpsc::Sender<Result<TranscriptionStatus, ServerFnError>>,
) -> miette::Result<()> {
    use futures::{SinkExt as _, StreamExt as _};
    let start_time = std::time::Instant::now();

    // Get the transcription request from the input stream
    let request = match input.next().await {
        Some(Ok(req)) => req,
        Some(Err(e)) => {
            return Err(miette::miette!(
                "Failed to receive transcription request: {}",
                e
            ));
        }
        None => {
            return Err(miette::miette!("No transcription request received"));
        }
    };

    // Send starting status
    if tx.send(Ok(TranscriptionStatus::Starting)).await.is_err() {
        return Ok(()); // Receiver dropped
    }

    info!("Starting transcription for file: {}", request.file_id);

    // Get file path and validate file exists
    let file_path = get_file_path(&request.file_id);

    if !file_path.exists() {
        error!("File not found: {:?}", file_path);
        return Err(miette::miette!("File not found: {:?}", file_path));
    }

    // Send processing audio status
    if tx
        .send(Ok(TranscriptionStatus::ProcessingAudio))
        .await
        .is_err()
    {
        return Ok(()); // Receiver dropped
    }

    info!("Processing audio file: {:?}", file_path);

    // First, get the actual audio duration
    let mut audio_processor = match AudioProcessor::new() {
        Ok(processor) => processor,
        Err(e) => {
            error!("Failed to create audio processor: {}", e);
            return Err(miette::miette!("Failed to create audio processor: {}", e));
        }
    };

    let audio_data = match audio_processor.load_audio(&file_path).await {
        Ok(data) => data,
        Err(e) => {
            error!("Failed to load audio file: {}", e);
            return Err(miette::miette!("Failed to load audio file: {}", e));
        }
    };

    let audio_duration = audio_data.duration;

    // Create transcription configuration from request
    let mut config = TranscriptionConfig::new().with_translate(request.translate);

    if let Some(language) = &request.language {
        config = config.with_language(language.clone());
    }

    // Start actual transcription streaming
    let streaming_result = match transcribe_file_stream(&file_path, Some(config)).await {
        Ok(result) => result,
        Err(e) => {
            error!("Failed to start transcription: {}", e);

            return Err(miette::miette!("Failed to start transcription: {}", e));
        }
    };

    let mut word_count = 0;
    let mut stream = streaming_result;

    // Process streaming chunks from purr-core
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(streaming_chunk) => {
                // Count words in this chunk
                let chunk_word_count = streaming_chunk.text.split_whitespace().count();
                word_count += chunk_word_count;

                // Send in-progress status with real chunk data
                if tx
                    .send(Ok(TranscriptionStatus::InProgress {
                        chunk_index: streaming_chunk.chunk_index,
                        text: streaming_chunk.text,
                        start_time: streaming_chunk.start,
                        end_time: streaming_chunk.end,
                    }))
                    .await
                    .is_err()
                {
                    return Ok(()); // Receiver dropped
                }

                // If this chunk contains final statistics, we're done
                if let Some(final_stats) = streaming_chunk.final_stats {
                    let processing_time = start_time.elapsed().as_secs_f64();

                    info!(
                        "Transcription completed for file: {} in {:.2}s",
                        request.file_id, processing_time
                    );

                    // Send completion status with real statistics
                    let _ = tx
                        .send(Ok(TranscriptionStatus::Completed {
                            processing_time,
                            audio_duration,
                            word_count: final_stats.word_count,
                        }))
                        .await;
                    return Ok(());
                }
            }
            Err(e) => {
                error!("Transcription chunk error: {}", e);
                return Err(miette::miette!("Transcription chunk error: {}", e));
            }
        }
    }

    // If we exit the loop without getting final stats, send completion with our counts
    let processing_time = start_time.elapsed().as_secs_f64();

    info!(
        "Transcription completed for file: {} in {:.2}s",
        request.file_id, processing_time
    );

    // Send completion status with real data
    let _ = tx
        .send(Ok(TranscriptionStatus::Completed {
            processing_time,
            audio_duration,
            word_count,
        }))
        .await;

    Ok(())
}
