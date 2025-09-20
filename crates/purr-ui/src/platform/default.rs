/// Default platform implementation for desktop/native builds
/// This implementation uses direct file system access and native whisper transcription
use super::{Platform, PlatformError, TranscriptionRequest, TranscriptionStatus};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use purr_common::platform::FileId;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use tracing::{error, info};

pub(super) struct PlatformImpl {
    temp_dir: PathBuf,
}

impl PlatformImpl {
    pub fn new() -> Self {
        let temp_dir = std::env::temp_dir().join("purr-temp");
        // Ensure temp directory exists
        let _ = std::fs::create_dir_all(&temp_dir);
        Self { temp_dir }
    }
}

#[async_trait::async_trait]
impl Platform for PlatformImpl {
    async fn process_file(
        &self,
        file_data: Bytes,
        file_path: &Path,
    ) -> Result<FileId, PlatformError> {
        use std::fs::File;
        use std::io::Write;
        use uuid::Uuid;

        // Generate unique file ID
        let file_id =
            Uuid::new_v3(&Uuid::NAMESPACE_URL, file_path.to_string_lossy().as_bytes()).into();
        let file_path = self.temp_dir.join(&file_id);

        info!(
            "Processing file {} with ID: {}",
            file_path.display(),
            file_id
        );

        // Write file data to temp location
        let mut file = File::create(&file_path).map_err(PlatformError::io)?;
        file.write_all(&file_data).map_err(PlatformError::io)?;
        file.flush().map_err(PlatformError::io)?;

        info!("File saved to: {:?}", file_path);
        Ok(file_id)
    }

    async fn transcribe(
        &self,
        file_id: FileId,
        request: TranscriptionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<TranscriptionStatus, PlatformError>> + Send>>,
        PlatformError,
    > {
        use futures::channel::mpsc;
        use futures::SinkExt;
        use purr_core::{transcribe_file_stream, TranscriptionConfig};

        let file_path = self.temp_dir.join(&file_id);

        // Create a channel for status updates
        let (mut tx, rx) = mpsc::channel(10);

        tokio::spawn(async move {
            let start_time = std::time::Instant::now();

            // Send starting status
            let _ = tx.send(Ok(TranscriptionStatus::Starting)).await;
            info!("Starting transcription for file: {}", file_id);

            // Create transcription configuration
            let mut config = TranscriptionConfig::new().with_translate(request.translate);
            if let Some(language) = request.language {
                config = config.with_language(language);
            }

            // Send processing audio status
            let _ = tx.send(Ok(TranscriptionStatus::ProcessingAudio)).await;

            // Start transcription
            match transcribe_file_stream(&file_path, Some(config)).await {
                Ok(mut stream) => {
                    let mut word_count = 0;
                    let mut audio_duration = 0.0f32;

                    // Process streaming chunks
                    while let Some(chunk_result) = stream.next().await {
                        match chunk_result {
                            Ok(chunk) => {
                                word_count += chunk.text.split_whitespace().count();

                                // Send progress update
                                let _ = tx
                                    .send(Ok(TranscriptionStatus::InProgress {
                                        chunk_index: chunk.chunk_index,
                                        text: chunk.text,
                                        start_time: chunk.start,
                                        end_time: chunk.end,
                                    }))
                                    .await;

                                // Check for final stats
                                if let Some(stats) = chunk.final_stats {
                                    audio_duration = stats.audio_duration;
                                    word_count = stats.word_count;
                                    break;
                                }
                            }
                            Err(e) => {
                                error!("Transcription error: {}", e);
                                let _ = tx.send(Err(PlatformError::transcription(e))).await;
                                return;
                            }
                        }
                    }

                    // Send completion status
                    let processing_time = start_time.elapsed().as_secs_f64();
                    let _ = tx
                        .send(Ok(TranscriptionStatus::Completed {
                            processing_time,
                            audio_duration,
                            word_count,
                        }))
                        .await;

                    info!("Transcription completed in {:.2}s", processing_time);
                }
                Err(e) => {
                    error!("Failed to start transcription: {}", e);
                    let _ = tx.send(Err(PlatformError::transcription(e))).await;
                }
            }
        });

        Ok(Box::pin(rx))
    }

    async fn cleanup(&self, file_id: &str) -> Result<(), PlatformError> {
        let file_path = self.temp_dir.join(file_id);
        if file_path.exists() {
            std::fs::remove_file(&file_path).map_err(PlatformError::io)?;
            info!("Cleaned up temporary file: {}", file_id);
        }
        Ok(())
    }
}
