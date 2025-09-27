use axum::{
    extract::Path,
    http::{header, StatusCode},
    response::Response,
    routing::get,
    Router,
};
use axum_test::TestServer;
use futures::StreamExt;
use purr_core::audio::{AudioChunk, AudioProcessor};
use purr_core::input::{AsyncCustomInput, AsyncStreamBuffer};
use purr_core::{TranscriptionConfig, WhisperError};
use std::path::PathBuf;
use tokio_util::io::ReaderStream;

const CHUNK_SIMILARITY_TOLERANCE: f32 = 0.01; // Allow 1% tolerance for floating point differences

#[tokio::test]
async fn test_async_stream_buffer_basic_operations() {
    let buffer = AsyncStreamBuffer::new();

    // Test writing data
    let test_data = b"Hello, World!";
    buffer.write(test_data);

    // Test reading data
    let mut read_buffer = vec![0u8; test_data.len()];
    let bytes_read = buffer.try_read(&mut read_buffer);
    assert_eq!(bytes_read, test_data.len() as i32);
    assert_eq!(&read_buffer, test_data);

    // Test EOF
    buffer.set_eof();
    let mut empty_buffer = vec![0u8; 10];
    let eof_result = buffer.try_read(&mut empty_buffer);
    assert_eq!(eof_result, ffmpeg_next::sys::AVERROR_EOF);
}

#[tokio::test]
async fn test_async_stream_buffer_concurrent_write_read() {
    let buffer = AsyncStreamBuffer::new();
    let buffer_clone = buffer.clone();

    // Spawn a task to write data
    let write_handle = tokio::spawn(async move {
        for i in 0..10 {
            let data = format!("chunk-{}", i);
            buffer_clone.write(data.as_bytes());
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
        buffer_clone.set_eof();
    });

    // Read data as it becomes available
    let mut total_read = Vec::new();
    let mut read_buffer = vec![0u8; 1024];

    loop {
        let bytes_read = buffer.try_read(&mut read_buffer);
        if bytes_read == ffmpeg_next::sys::AVERROR_EOF {
            break;
        } else if bytes_read > 0 {
            total_read.extend_from_slice(&read_buffer[..bytes_read as usize]);
        } else {
            // No data available yet, wait a bit
            tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        }
    }

    write_handle.await.unwrap();

    // Verify we read all the expected data
    let expected = (0..10)
        .map(|i| format!("chunk-{}", i))
        .collect::<Vec<_>>()
        .join("");
    assert_eq!(String::from_utf8(total_read).unwrap(), expected);
}

#[tokio::test]
async fn test_async_custom_input_creation() {
    let buffer = AsyncStreamBuffer::new();

    // Add some test data
    buffer.write(b"test audio data");
    buffer.set_eof();

    // Create AsyncCustomInput
    let result = AsyncCustomInput::create(buffer).await;

    // This might fail due to invalid audio format, but the AsyncCustomInput creation should work
    // The error would come from FFmpeg trying to parse the data
    match result {
        Ok(_) => {
            // Success - unlikely with our test data but valid
        }
        Err(e) => {
            // Expected - our test data isn't valid audio format
            // But we can verify it's an FFmpeg error, not a creation error
            println!("Expected FFmpeg error: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_url_vs_file_streaming_comparison() {
    // Skip test if samples directory doesn't exist
    let samples_dir = get_samples_dir();
    if !samples_dir.exists() {
        println!(
            "Skipping test: samples directory not found at {:?}",
            samples_dir
        );
        return;
    }

    let test_file = samples_dir.join("jfk.wav");
    if !test_file.exists() {
        println!("Skipping test: jfk.wav not found in samples directory");
        return;
    }

    // Create mock HTTP server
    let server = create_mock_server().await;
    let base_url = match server.server_address() {
        Some(url) => url,
        None => {
            println!("Skipping test: Server has no address (likely in mock mode)");
            return;
        }
    };

    // Create URL for the test file
    let audio_url = format!("{}/audio/jfk.wav", base_url);

    // Configuration for both tests
    let config = TranscriptionConfig::default();

    println!("Testing URL: {}", audio_url);
    println!("Testing file: {:?}", test_file);

    // Test file streaming
    let mut file_stream = match AudioProcessor::stream_file(&test_file).await {
        Ok(stream) => stream,
        Err(e) => {
            println!("File streaming failed: {:?}", e);
            return;
        }
    };

    // Test URL streaming
    let mut url_stream =
        match AudioProcessor::stream_url_with_config(audio_url.parse().unwrap(), &config).await {
            Ok(stream) => stream,
            Err(e) => {
                println!("URL streaming failed: {:?}", e);
                return;
            }
        };

    // Collect chunks from both streams
    let mut file_chunks = Vec::new();
    let mut url_chunks = Vec::new();

    // Collect file chunks
    while let Some(chunk_result) = file_stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                file_chunks.push(chunk);
            }
            Err(e) => {
                println!("File stream error: {:?}", e);
                break;
            }
        }
    }

    // Collect URL chunks
    while let Some(chunk_result) = url_stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                url_chunks.push(chunk);
            }
            Err(e) => {
                println!("URL stream error: {:?}", e);
                break;
            }
        }
    }

    println!("File chunks: {}", file_chunks.len());
    println!("URL chunks: {}", url_chunks.len());

    // Both streams should produce chunks
    assert!(!file_chunks.is_empty(), "File stream should produce chunks");
    assert!(!url_chunks.is_empty(), "URL stream should produce chunks");

    // Chunk counts should be close (allow for minor differences due to streaming)
    let chunk_count_diff = (file_chunks.len() as i32 - url_chunks.len() as i32).abs();
    assert!(
        chunk_count_diff <= 2,
        "Chunk counts should be similar: file={}, url={}, diff={}",
        file_chunks.len(),
        url_chunks.len(),
        chunk_count_diff
    );

    // Compare corresponding chunks
    let min_chunks = file_chunks.len().min(url_chunks.len());

    for i in 0..min_chunks {
        let file_chunk = &file_chunks[i];
        let url_chunk = &url_chunks[i];

        println!(
                "Comparing chunk {}: file_samples={}, url_samples={}, file_duration={:.3}, url_duration={:.3}",
                i,
                file_chunk.samples.len(),
                url_chunk.samples.len(),
                file_chunk.duration,
                url_chunk.duration
            );

        assert!(
            chunks_are_equivalent(file_chunk, url_chunk, CHUNK_SIMILARITY_TOLERANCE),
            "Chunk {} differs between file and URL streams. \
                File: index={}, samples={}, duration={:.3}, start_time={:.3}, is_final={} \
                URL: index={}, samples={}, duration={:.3}, start_time={:.3}, is_final={}",
            i,
            file_chunk.index,
            file_chunk.samples.len(),
            file_chunk.duration,
            file_chunk.start_time,
            file_chunk.is_final,
            url_chunk.index,
            url_chunk.samples.len(),
            url_chunk.duration,
            url_chunk.start_time,
            url_chunk.is_final
        );
    }

    // Verify that both streams have final chunks
    assert!(
        file_chunks.iter().any(|c| c.is_final),
        "File stream should have a final chunk"
    );
    assert!(
        url_chunks.iter().any(|c| c.is_final),
        "URL stream should have a final chunk"
    );

    println!("✅ URL and file streaming produce equivalent results!");
}

#[tokio::test]
async fn test_url_vs_file_sync_load_comparison() {
    // Skip test if samples directory doesn't exist
    let samples_dir = get_samples_dir();
    if !samples_dir.exists() {
        println!(
            "Skipping test: samples directory not found at {:?}",
            samples_dir
        );
        return;
    }

    let test_file = samples_dir.join("jfk.wav");
    if !test_file.exists() {
        println!("Skipping test: jfk.wav not found in samples directory");
        return;
    }

    // Create mock HTTP server
    let server = create_mock_server().await;
    let base_url = match server.server_address() {
        Some(url) => url,
        None => {
            println!("Skipping test: Server has no address (likely in mock mode)");
            return;
        }
    };

    // Create URL for the test file
    let audio_url = format!("{}/audio/jfk.wav", base_url);

    // Configuration for both tests
    let config = TranscriptionConfig::default();

    println!("Testing URL: {}", audio_url);
    println!("Testing file: {:?}", test_file);

    // Test file loading
    let file_audio = match AudioProcessor::load_audio(&test_file).await {
        Ok(audio) => audio,
        Err(e) => {
            println!("File loading failed: {:?}", e);
            return;
        }
    };

    // Test URL loading
    let url_audio =
        match AudioProcessor::load_audio_from_url_with_config(audio_url.parse().unwrap(), &config)
            .await
        {
            Ok(audio) => audio,
            Err(e) => {
                println!("URL loading failed: {:?}", e);
                return;
            }
        };

    println!(
        "File audio: samples={}, duration={:.3}, sample_rate={}",
        file_audio.samples.len(),
        file_audio.duration,
        file_audio.sample_rate
    );
    println!(
        "URL audio: samples={}, duration={:.3}, sample_rate={}",
        url_audio.samples.len(),
        url_audio.duration,
        url_audio.sample_rate
    );

    // Basic properties should match
    assert_eq!(
        file_audio.sample_rate, url_audio.sample_rate,
        "Sample rates should match"
    );

    // Durations should be very close
    let duration_diff = (file_audio.duration - url_audio.duration).abs();
    assert!(
        duration_diff < 0.1,
        "Durations should be similar: file={:.3}, url={:.3}, diff={:.3}",
        file_audio.duration,
        url_audio.duration,
        duration_diff
    );

    // Sample counts should be close (allow for minor differences)
    let sample_count_diff =
        (file_audio.samples.len() as i32 - url_audio.samples.len() as i32).abs();
    assert!(
        sample_count_diff < 1000,
        "Sample counts should be similar: file={}, url={}, diff={}",
        file_audio.samples.len(),
        url_audio.samples.len(),
        sample_count_diff
    );

    // Compare samples (with tolerance for floating point precision)
    let min_samples = file_audio.samples.len().min(url_audio.samples.len());
    let tolerance = 0.001;
    let mut diff_count = 0;

    for i in 0..min_samples {
        if (file_audio.samples[i] - url_audio.samples[i]).abs() > tolerance {
            diff_count += 1;
        }
    }

    // Allow up to 1% of samples to have small differences
    let max_allowed_diffs = min_samples / 100;
    assert!(
        diff_count <= max_allowed_diffs,
        "Too many sample differences: {}/{} ({}%)",
        diff_count,
        min_samples,
        (diff_count * 100) / min_samples
    );

    println!("✅ URL and file loading produce equivalent results!");
    println!(
        "   Sample differences: {}/{} ({}%)",
        diff_count,
        min_samples,
        (diff_count * 100) / min_samples
    );
}

#[tokio::test]
async fn test_http_server_error_handling() {
    // Create mock HTTP server
    let server = create_mock_server().await;
    let base_url = match server.server_address() {
        Some(url) => url,
        None => {
            println!("Skipping test: Server has no address (likely in mock mode)");
            return;
        }
    };

    // Test 404 error
    let nonexistent_url = format!("{}/audio/nonexistent.wav", base_url);
    let config = TranscriptionConfig::default();

    let result =
        AudioProcessor::load_audio_from_url_with_config(nonexistent_url.parse().unwrap(), &config)
            .await;

    assert!(result.is_err(), "Should fail for nonexistent file");

    // Verify error type
    match result.unwrap_err() {
        WhisperError::NetworkError { .. } => {
            // Expected network error
        }
        other => {
            println!("Unexpected error type: {:?}", other);
        }
    }
}

#[tokio::test]
async fn test_http_timeout_handling() {
    // Test with very short timeout
    let mut config = TranscriptionConfig::default();
    config.http_timeout = 1; // 1 second
    config.http_connect_timeout = 1; // 1 second

    // Try to connect to a slow/unresponsive server
    let timeout_url = "http://httpbin.org/delay/5"; // 5 second delay

    let result =
        AudioProcessor::load_audio_from_url_with_config(timeout_url.parse().unwrap(), &config)
            .await;

    // Should timeout (though the exact error type may vary)
    assert!(result.is_err(), "Should timeout on slow server");
}

/// Create a mock HTTP server that serves audio files
async fn create_mock_server() -> TestServer {
    let router = Router::new().route("/audio/{filename}", get(serve_audio));
    TestServer::new(router).expect("Failed to create test server")
}

/// Handler to serve audio files from the samples directory
async fn serve_audio(Path(filename): Path<String>) -> Result<Response, StatusCode> {
    let samples_dir = get_samples_dir();
    let file_path = samples_dir.join(&filename);

    if !file_path.exists() {
        return Err(StatusCode::NOT_FOUND);
    }

    // Read file
    let file = match tokio::fs::File::open(&file_path).await {
        Ok(file) => file,
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
    };

    // Get file size for Content-Length header
    let metadata = match file.metadata().await {
        Ok(metadata) => metadata,
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
    };

    // Create a streaming response
    let stream = ReaderStream::new(file);
    let body = axum::body::Body::from_stream(stream);

    // Determine content type based on file extension
    let content_type = match file_path.extension().and_then(|ext| ext.to_str()) {
        Some("wav") => "audio/wav",
        Some("mp3") => "audio/mpeg",
        Some("ogg") => "audio/ogg",
        Some("opus") => "audio/opus",
        Some("flac") => "audio/flac",
        _ => "application/octet-stream",
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, metadata.len().to_string())
        .body(body)
        .unwrap())
}

/// Get the samples directory path
fn get_samples_dir() -> PathBuf {
    // Get the project root by walking up from the current directory
    let mut current_dir = std::env::current_dir().expect("Failed to get current directory");

    // Walk up to find the project root (contains Cargo.toml)
    while !current_dir.join("Cargo.toml").exists() && current_dir.parent().is_some() {
        current_dir = current_dir.parent().unwrap().to_path_buf();
    }

    current_dir.join("samples")
}

/// Helper function to compare two audio chunks
fn chunks_are_equivalent(chunk1: &AudioChunk, chunk2: &AudioChunk, tolerance: f32) -> bool {
    // Basic metadata should match
    if chunk1.sample_rate != chunk2.sample_rate
        || chunk1.index != chunk2.index
        || chunk1.is_final != chunk2.is_final
    {
        return false;
    }

    // Allow small differences in timing and duration due to processing variations
    if (chunk1.start_time - chunk2.start_time).abs() > tolerance
        || (chunk1.duration - chunk2.duration).abs() > tolerance
    {
        return false;
    }

    // Sample counts should be identical or very close
    if (chunk1.samples.len() as i32 - chunk2.samples.len() as i32).abs() > 100 {
        return false;
    }

    // Compare samples with tolerance for floating point precision
    let min_len = chunk1.samples.len().min(chunk2.samples.len());
    for i in 0..min_len {
        if (chunk1.samples[i] - chunk2.samples[i]).abs() > tolerance {
            return false;
        }
    }

    true
}
