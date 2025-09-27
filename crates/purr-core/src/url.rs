use crate::error::{Result, WhisperError};
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
    /// HTTP timeout for requests
    pub timeout: Duration,
    /// HTTP connection timeout
    pub connect_timeout: Duration,
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
            timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            user_agent: format!("purr-audio-transcriber/{}", env!("CARGO_PKG_VERSION")),
            max_redirects: 10,
            buffer_size: 8192,
        }
    }
}

/// Validates if a string is a valid URL
pub fn is_valid_url(input: &str) -> bool {
    match Url::parse(input) {
        Ok(url) => {
            // Check if it's a HTTP or HTTPS URL
            matches!(url.scheme(), "http" | "https")
        }
        Err(_) => false,
    }
}

/// Validates if a string looks like a file path
pub fn is_file_path(input: &str) -> bool {
    // Consider it a file path if:
    // 1. It doesn't start with http:// or https://
    // 2. It contains path separators or is a simple filename
    !input.starts_with("http://")
        && !input.starts_with("https://")
        && (input.contains('/')
            || input.contains('\\')
            || !input.contains('.')
            || input.split('.').count() <= 2)
}

/// HTTP streaming client for audio URLs
pub struct HttpStreamer {
    client: Client,
    config: UrlStreamConfig,
}

impl HttpStreamer {
    /// Create a new URL streamer with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(UrlStreamConfig::default())
    }

    /// Create a new URL streamer with custom configuration
    pub fn with_config(config: UrlStreamConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .user_agent(&config.user_agent)
            .redirect(reqwest::redirect::Policy::limited(config.max_redirects))
            .build()
            .map_err(|e| WhisperError::NetworkError {
                message: format!("Failed to create HTTP client: {}", e),
            })?;

        Ok(Self { client, config })
    }

    /// Stream audio data from a URL into an AsyncStreamBuffer
    pub async fn stream_url(
        &self,
        url: Url,
        buffer: Arc<AsyncStreamBuffer>,
    ) -> Result<JoinHandle<Result<()>>> {
        if !matches!(url.scheme(), "http" | "https") {
            return Err(WhisperError::InvalidInput {
                message: "Only HTTP and HTTPS URLs are supported".to_string(),
            });
        }

        info!("Starting to stream audio from URL: {}", url);

        // Start the streaming request
        let response =
            self.client
                .get(url.clone())
                .send()
                .await
                .map_err(|e| WhisperError::NetworkError {
                    message: format!("Failed to connect to URL: {}", e),
                })?;

        // Check response status
        if !response.status().is_success() {
            return Err(WhisperError::NetworkError {
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

        // Clone necessary data for the async task
        let buffer_clone = buffer.clone();
        let _config = self.config.clone();

        // Spawn async task to stream the response
        let handle = tokio::spawn(async move {
            let mut stream = response.bytes_stream();
            let mut total_received = 0u64;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        total_received += chunk.len() as u64;
                        debug!(
                            "Received chunk of {} bytes (total: {} bytes)",
                            chunk.len(),
                            total_received
                        );

                        // Write chunk to buffer
                        buffer_clone.write(&chunk);
                    }
                    Err(e) => {
                        error!("Error receiving chunk: {}", e);
                        return Err(WhisperError::NetworkError {
                            message: format!("Error receiving data chunk: {}", e),
                        });
                    }
                }
            }

            info!("Finished streaming {} bytes from URL", total_received);
            buffer_clone.set_eof();
            Ok(())
        });

        Ok(handle)
    }

    /// Get the content length of a URL without downloading it
    pub async fn get_content_info(&self, url: &str) -> Result<UrlContentInfo> {
        let url = Url::parse(url).map_err(|e| WhisperError::InvalidInput {
            message: format!("Invalid URL: {}", e),
        })?;

        let response =
            self.client
                .head(url.clone())
                .send()
                .await
                .map_err(|e| WhisperError::NetworkError {
                    message: format!("Failed to get URL info: {}", e),
                })?;

        if !response.status().is_success() {
            return Err(WhisperError::NetworkError {
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
        self.content_type
            .as_ref()
            .map(|ct| {
                ct.starts_with("audio/")
                    || ct.starts_with("video/")
                    || ct == "application/octet-stream"
            })
            .unwrap_or(true) // If no content type, assume it might be audio
    }

    /// Get a human-readable size string
    pub fn size_string(&self) -> String {
        match self.content_length {
            Some(size) => format_bytes(size),
            None => "Unknown size".to_string(),
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
    fn test_is_valid_url() {
        assert!(is_valid_url("https://example.com/audio.mp3"));
        assert!(is_valid_url("http://example.com/audio.wav"));
        assert!(!is_valid_url("ftp://example.com/file.mp3"));
        assert!(!is_valid_url("file.mp3"));
        assert!(!is_valid_url("/path/to/file.mp3"));
        assert!(!is_valid_url("not-a-url"));
    }

    #[test]
    fn test_is_file_path() {
        assert!(is_file_path("file.mp3"));
        assert!(is_file_path("/path/to/file.mp3"));
        assert!(is_file_path("./relative/path.wav"));
        assert!(is_file_path("../file.ogg"));
        assert!(!is_file_path("https://example.com/audio.mp3"));
        assert!(!is_file_path("http://example.com/audio.wav"));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(1073741824), "1.0 GB");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.5 KB");
    }
}
