use bytes::Bytes;
use dioxus::prelude::*;
#[cfg(feature = "server")]
use dioxus_fullstack::BoxedStream;
use dioxus_fullstack::{
    codec::JsonEncoding, ContentType, Decodes, Encodes, Format, FormatType, Websocket,
};
use futures::{channel::mpsc, SinkExt, StreamExt};
use serde::{Deserialize, Serialize};

#[server(protocol = Websocket<BytesEncoding, JsonEncoding>)]
pub async fn upload_file(
    input: BoxedStream<Bytes, ServerFnError>,
) -> ServerFnResult<BoxedStream<UploadStatus, ServerFnError>> {
    let mut input = input;

    // Create a channel with the output of the websocket
    let (mut tx, rx) = mpsc::channel(10);
    tokio::spawn(async move {
        let mut total_bytes = 0;
        while let Some(msg) = input.next().await {
            match msg {
                Ok(bytes) => {
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
        let _ = tx.send(Ok(UploadStatus::Completed { total_bytes })).await;
    });

    Ok(rx.into())
}

#[derive(Serialize, Deserialize, Debug)]
pub enum UploadStatus {
    InProgress { bytes_received: usize },
    Completed { total_bytes: usize },
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
