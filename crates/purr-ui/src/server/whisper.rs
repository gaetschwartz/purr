use dioxus::prelude::*;
use dioxus_fullstack::{codec::JsonEncoding, BoxedStream, Websocket};
use serde::{Deserialize, Serialize};

#[cfg(feature = "server")]
use {
    std::{env, path::PathBuf},
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
#[cfg(feature = "server")]
fn get_file_path(file_id: &str) -> PathBuf {
    env::temp_dir().join("purr-uploads").join(file_id)
}

/// Start transcription with streaming status updates
#[server(protocol = Websocket<JsonEncoding, JsonEncoding>)]
pub async fn start_transcription(
    input: BoxedStream<TranscriptionRequest, ServerFnError>,
) -> ServerFnResult<BoxedStream<TranscriptionStatus, ServerFnError>> {
    use futures::{channel::mpsc, SinkExt as _, StreamExt as _};
    let mut input = input;

    // Create a channel for status updates
    let (mut tx, rx) = mpsc::channel(10);

    tokio::spawn(async move {
        let start_time = std::time::Instant::now();

        // Get the transcription request from the input stream
        let request = match input.next().await {
            Some(Ok(req)) => req,
            Some(Err(e)) => {
                let _ = tx.send(Err(e)).await;
                return;
            }
            None => {
                let _ = tx
                    .send(Ok(TranscriptionStatus::Error {
                        message: "No transcription request received".to_string(),
                    }))
                    .await;
                return;
            }
        };

        // Send starting status
        if tx.send(Ok(TranscriptionStatus::Starting)).await.is_err() {
            return; // Receiver dropped
        }

        info!("Starting transcription for file: {}", request.file_id);

        // Get file path and validate file exists
        let file_path = get_file_path(&request.file_id);

        if !file_path.exists() {
            error!("File not found: {:?}", file_path);
            let _ = tx
                .send(Ok(TranscriptionStatus::Error {
                    message: format!("File not found: {}", request.file_id),
                }))
                .await;
            return;
        }

        // Send processing audio status
        if tx
            .send(Ok(TranscriptionStatus::ProcessingAudio))
            .await
            .is_err()
        {
            return;
        }

        info!("Processing audio file: {:?}", file_path);

        // Simulate audio processing time
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Simulate transcription chunks - in real implementation, this would be
        // replaced with actual whisper transcription that yields chunks
        let mock_transcription_chunks = [
            ("Hello, this is a test transcription.", 0.0, 2.5),
            ("The audio quality seems good.", 2.5, 5.0),
            ("Transcription is working as expected.", 5.0, 8.2),
            ("This is the final chunk of text.", 8.2, 10.5),
        ];

        let mut word_count = 0;

        for (chunk_index, (text, start_time, end_time)) in
            mock_transcription_chunks.iter().enumerate()
        {
            // Count words in this chunk
            word_count += text.split_whitespace().count();

            // Send in-progress status with chunk
            if tx
                .send(Ok(TranscriptionStatus::InProgress {
                    chunk_index,
                    text: text.to_string(),
                    start_time: *start_time,
                    end_time: *end_time,
                }))
                .await
                .is_err()
            {
                return; // Receiver dropped
            }

            // Simulate processing time between chunks
            tokio::time::sleep(tokio::time::Duration::from_millis(800)).await;
        }

        let processing_time = start_time.elapsed().as_secs_f64();
        let audio_duration = 10.5; // Mock audio duration

        info!(
            "Transcription completed for file: {} in {:.2}s",
            request.file_id, processing_time
        );

        // Send completion status
        let _ = tx
            .send(Ok(TranscriptionStatus::Completed {
                processing_time,
                audio_duration,
                word_count,
            }))
            .await;
    });

    Ok(rx.into())
}
