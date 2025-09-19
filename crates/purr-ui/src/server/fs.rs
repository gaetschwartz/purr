use bytes::Bytes;
use dioxus::prelude::*;
use dioxus_fullstack::{
    codec::JsonEncoding, BoxedStream, ContentType, Decodes, Encodes, Format, FormatType, Websocket,
};
use serde::{Deserialize, Serialize};
#[cfg(feature = "server")]
use {
    std::{env, fs::File, io::Write},
    tracing::{error, info},
    uuid::Uuid,
};

#[server(protocol = Websocket<BytesEncoding, JsonEncoding>)]
pub async fn upload_file(
    input: BoxedStream<Bytes, ServerFnError>,
) -> ServerFnResult<BoxedStream<UploadStatus, ServerFnError>> {
    use futures::{channel::mpsc, SinkExt as _, StreamExt as _};
    let mut input = input;

    // Create a channel with the output of the websocket
    let (mut tx, rx) = mpsc::channel(10);
    tokio::spawn(async move {
        // Generate unique file ID and create temp file path
        let file_id = Uuid::new_v4().to_string();
        let temp_dir = env::temp_dir().join("purr-uploads");

        // Ensure temp directory exists
        if let Err(e) = std::fs::create_dir_all(&temp_dir) {
            error!("Failed to create temp directory: {}", e);
            let _ = tx
                .send(Err(ServerFnError::new(format!(
                    "Failed to create temp directory: {}",
                    e
                ))))
                .await;
            return;
        }

        let file_path = temp_dir.join(&file_id);
        let mut file = match File::create(&file_path) {
            Ok(f) => f,
            Err(e) => {
                error!("Failed to create file: {}", e);
                let _ = tx
                    .send(Err(ServerFnError::new(format!(
                        "Failed to create file: {}",
                        e
                    ))))
                    .await;
                return;
            }
        };

        let mut total_bytes = 0;
        info!("Starting file upload with ID: {}", file_id);

        while let Some(msg) = input.next().await {
            match msg {
                Ok(bytes) => {
                    // Write bytes to file
                    if let Err(e) = file.write_all(&bytes) {
                        error!("Failed to write to file: {}", e);
                        let _ = tx
                            .send(Err(ServerFnError::new(format!(
                                "Failed to write to file: {}",
                                e
                            ))))
                            .await;
                        return;
                    }

                    total_bytes += bytes.len();
                    if tx
                        .send(Ok(UploadStatus::InProgress {
                            bytes_received: total_bytes,
                        }))
                        .await
                        .is_err()
                    {
                        // Receiver dropped
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            }
        }

        // Flush and sync file
        if let Err(e) = file.flush() {
            error!("Failed to flush file: {}", e);
            let _ = tx
                .send(Err(ServerFnError::new(format!(
                    "Failed to flush file: {}",
                    e
                ))))
                .await;
            return;
        }

        info!("File upload completed: {} ({} bytes)", file_id, total_bytes);
        let _ = tx
            .send(Ok(UploadStatus::Completed {
                total_bytes,
                file_id,
            }))
            .await;
    });

    Ok(rx.into())
}

#[derive(Serialize, Deserialize, Debug)]
pub enum UploadStatus {
    InProgress { bytes_received: usize },
    Completed { total_bytes: usize, file_id: String },
}

pub struct BytesEncoding;

impl Encodes<Bytes> for BytesEncoding {
    type Error = std::convert::Infallible;

    fn encode(output: &Bytes) -> Result<Bytes, Self::Error> {
        Ok(output.clone())
    }
}

impl Decodes<Bytes> for BytesEncoding {
    type Error = std::convert::Infallible;

    fn decode(bytes: Bytes) -> Result<Bytes, Self::Error> {
        Ok(bytes)
    }
}

impl FormatType for BytesEncoding {
    const FORMAT_TYPE: Format = Format::Binary;
}

impl ContentType for BytesEncoding {
    const CONTENT_TYPE: &'static str = "application/octet-stream";
}
