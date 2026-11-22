use crate::error::{Error, Result};
use crate::input::AsyncStreamBuffer;
use futures_util::StreamExt;
use reqwest::Client;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};
use url::Url;

/// Configuration for URL streaming
#[derive(Debug, Clone)]
pub struct UrlStreamConfig {
    /// HTTP connection timeout
    pub connect_timeout: Duration,
    /// How long a read may wait for data before the transfer fails
    pub read_timeout: Duration,
    /// User agent string for requests
    pub user_agent: String,
    /// Maximum number of redirects to follow
    pub max_redirects: usize,
    /// Buffer size for streaming chunks
    pub buffer_size: usize,
}

impl Default for UrlStreamConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(10),
            read_timeout: Duration::from_secs(30),
            user_agent: format!("purr-audio-transcriber/{}", env!("CARGO_PKG_VERSION")),
            max_redirects: 10,
            buffer_size: 8192,
        }
    }
}

/// A download feeding a buffer in the background.
pub struct StreamingDownload {
    pub task: JoinHandle<Result<()>>,
    /// The size of the body, when the server announced it
    pub content_length: Option<u64>,
}

fn describe(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}

/// HTTP streaming client for audio URLs
pub struct HttpStreamer {
    client: Client,
    read_timeout: Duration,
}

impl HttpStreamer {
    /// Create a new URL streamer with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(UrlStreamConfig::default())
    }

    /// Create a new URL streamer with custom configuration
    pub fn with_config(config: UrlStreamConfig) -> Result<Self> {
        let client = crate::http::client_builder()?
            .connect_timeout(config.connect_timeout)
            .user_agent(&config.user_agent)
            .redirect(reqwest::redirect::Policy::limited(config.max_redirects))
            .build()
            .map_err(|e| Error::NetworkError {
                message: format!("Failed to create HTTP client: {e}"),
            })?;

        Ok(Self {
            client,
            read_timeout: config.read_timeout,
        })
    }

    /// Stream audio data from a URL into an AsyncStreamBuffer
    ///
    /// The buffer ends with `set_eof` when the body ends and with `fail` when it breaks off.
    pub async fn stream_url(
        &self,
        url: Url,
        buffer: Arc<AsyncStreamBuffer>,
    ) -> Result<StreamingDownload> {
        if !matches!(url.scheme(), "http" | "https") {
            return Err(Error::InvalidInput {
                message: "Only HTTP and HTTPS URLs are supported".to_owned(),
            });
        }

        info!("Starting to stream audio from URL: {}", url);

        // Start the streaming request
        let response = self
            .within(self.client.get(url.clone()).send())
            .await?
            .map_err(|e| Error::NetworkError {
                message: format!("Failed to connect to URL: {e}"),
            })?;

        // Check response status
        if !response.status().is_success() {
            return Err(Error::NetworkError {
                message: format!(
                    "HTTP request failed with status: {} for URL: {}",
                    response.status(),
                    url
                ),
            });
        }

        // Get content length if available
        let content_length = response.content_length();
        if let Some(length) = content_length {
            debug!("Content length: {} bytes", length);
        } else {
            debug!("Content length unknown (chunked transfer)");
        }

        // Validate content type if available
        if let Some(content_type) = response.headers().get("content-type") {
            let content_type_str = content_type.to_str().unwrap_or("");
            debug!("Content type: {}", content_type_str);

            // Warn if content type doesn't look like audio
            if !content_type_str.starts_with("audio/")
                && !content_type_str.starts_with("application/octet-stream")
                && !content_type_str.starts_with("video/")
            {
                warn!(
                    "Content type '{}' may not be an audio file",
                    content_type_str
                );
            }
        }

        let read_timeout = self.read_timeout;
        let task = tokio::spawn(async move {
            let mut stream = response.bytes_stream();
            let mut total_received = 0u64;

            loop {
                let next = match tokio::time::timeout(read_timeout, stream.next()).await {
                    Ok(next) => next,
                    Err(elapsed) => {
                        let message = format!(
                            "No data received for {} seconds: {elapsed}",
                            read_timeout.as_secs()
                        );
                        error!("{message}");
                        buffer.fail(message.clone());
                        return Err(Error::NetworkError { message });
                    }
                };
                let Some(chunk_result) = next else {
                    break;
                };
                match chunk_result {
                    Ok(chunk) => {
                        total_received += chunk.len() as u64;
                        debug!(
                            "Received chunk of {} bytes (total: {} bytes)",
                            chunk.len(),
                            total_received
                        );
                        if !buffer.push(&chunk).await {
                            debug!("The reader is gone, stopping the download");
                            return Ok(());
                        }
                    }
                    Err(e) => {
                        let message = format!("Error receiving data chunk: {}", describe(&e));
                        error!("{message}");
                        buffer.fail(message.clone());
                        return Err(Error::NetworkError { message });
                    }
                }
            }

            info!("Finished streaming {} bytes from URL", total_received);
            buffer.set_eof();
            Ok(())
        });

        Ok(StreamingDownload {
            task,
            content_length,
        })
    }

    /// Awaits `request`, which must answer within the read timeout.
    ///
    /// The timeout runs only while waiting on the network, so a consumer that reads slowly
    /// never trips it; the client's own read timeout would count that time too.
    async fn within<T>(&self, request: impl Future<Output = T>) -> Result<T> {
        tokio::time::timeout(self.read_timeout, request)
            .await
            .map_err(|elapsed| Error::NetworkError {
                message: format!(
                    "No response within {} seconds: {elapsed}",
                    self.read_timeout.as_secs()
                ),
            })
    }

    /// Get the content length of a URL without downloading it
    pub async fn get_content_info(&self, url: &str) -> Result<UrlContentInfo> {
        let url = Url::parse(url).map_err(|e| Error::InvalidInput {
            message: format!("Invalid URL: {e}"),
        })?;

        let response = self
            .within(self.client.head(url.clone()).send())
            .await?
            .map_err(|e| Error::NetworkError {
                message: format!("Failed to get URL info: {e}"),
            })?;

        if !response.status().is_success() {
            return Err(Error::NetworkError {
                message: format!(
                    "HEAD request failed with status: {} for URL: {}",
                    response.status(),
                    url
                ),
            });
        }

        let content_length = response.content_length();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|ct| ct.to_str().ok())
            .map(String::from);

        Ok(UrlContentInfo {
            content_length,
            content_type,
            url: url.to_string(),
        })
    }
}

/// Information about URL content
#[derive(Debug, Clone)]
pub struct UrlContentInfo {
    pub content_length: Option<u64>,
    pub content_type: Option<String>,
    pub url: String,
}

impl UrlContentInfo {
    /// Check if the content type suggests this is an audio file
    pub fn is_likely_audio(&self) -> bool {
        self.content_type.as_ref().is_none_or(|ct| {
            ct.starts_with("audio/") || ct.starts_with("video/") || ct == "application/octet-stream"
        }) // If no content type, assume it might be audio
    }

    /// Get a human-readable size string
    pub fn size_string(&self) -> String {
        match self.content_length {
            Some(size) => format_bytes(size),
            None => "Unknown size".to_owned(),
        }
    }
}

/// Format bytes in human-readable format
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", size as u64, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(1073741824), "1.0 GB");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.5 KB");
    }
}
