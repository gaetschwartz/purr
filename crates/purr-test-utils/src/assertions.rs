//! Custom assertion macros for Purr-specific testing patterns

/// Assert that an async operation completes within a timeout
#[macro_export]
macro_rules! assert_async_timeout {
    ($timeout_ms:expr, $fut:expr) => {
        match tokio::time::timeout(std::time::Duration::from_millis($timeout_ms), $fut).await {
            Ok(result) => result,
            Err(_) => panic!("Operation timed out after {}ms", $timeout_ms),
        }
    };
}

/// Assert that a WebGPU operation succeeds
#[cfg(target_arch = "wasm32")]
#[macro_export]
macro_rules! assert_webgpu_ok {
    ($result:expr) => {
        match $result {
            Ok(val) => val,
            Err(e) => panic!("WebGPU operation failed: {:?}", e),
        }
    };
    ($result:expr, $msg:expr) => {
        match $result {
            Ok(val) => val,
            Err(e) => panic!("{}: {:?}", $msg, e),
        }
    };
}

/// Assert that audio data has expected properties
#[macro_export]
macro_rules! assert_audio_valid {
    ($audio_data:expr, $sample_rate:expr) => {
        assert!(!$audio_data.is_empty(), "Audio data should not be empty");
        assert_eq!(
            $audio_data.len() % $sample_rate as usize,
            0,
            "Audio data length should be divisible by sample rate"
        );
    };
    ($audio_data:expr, $sample_rate:expr, $channels:expr) => {
        assert!(!$audio_data.is_empty(), "Audio data should not be empty");
        assert_eq!(
            $audio_data.len() % ($sample_rate as usize * $channels),
            0,
            "Audio data length should match sample rate and channel count"
        );
    };
}

/// Assert that transcription result has expected structure
#[macro_export]
macro_rules! assert_transcription_valid {
    ($transcription:expr) => {
        assert!(
            !$transcription.text.trim().is_empty(),
            "Transcription text should not be empty"
        );
        assert!(
            $transcription.confidence >= 0.0 && $transcription.confidence <= 1.0,
            "Confidence should be between 0.0 and 1.0, got: {}",
            $transcription.confidence
        );
        assert!(
            $transcription.duration > std::time::Duration::ZERO,
            "Duration should be positive"
        );
    };
}

/// Assert that a file exists and has expected size
#[macro_export]
macro_rules! assert_file_size_range {
    ($path:expr, $min_bytes:expr, $max_bytes:expr) => {
        let metadata = std::fs::metadata($path)
            .unwrap_or_else(|_| panic!("File should exist: {}", $path.display()));
        let size = metadata.len();
        assert!(
            size >= $min_bytes && size <= $max_bytes,
            "File size {} should be between {} and {} bytes",
            size,
            $min_bytes,
            $max_bytes
        );
    };
}

/// Assert that memory usage is within acceptable bounds
#[macro_export]
macro_rules! assert_memory_usage {
    ($max_mb:expr, $closure:expr) => {
        let start_memory = $crate::get_memory_usage();
        $closure;
        let end_memory = $crate::get_memory_usage();
        let used_mb = (end_memory - start_memory) / 1024 / 1024;
        assert!(
            used_mb <= $max_mb,
            "Memory usage {} MB exceeded limit {} MB",
            used_mb,
            $max_mb
        );
    };
}

/// Assert that concurrent operations maintain consistency
#[macro_export]
macro_rules! assert_concurrent_consistency {
    ($initial_state:expr, $operations:expr, $validator:expr) => {
        let state = std::sync::Arc::new(std::sync::Mutex::new($initial_state));
        let handles: Vec<_> = $operations
            .into_iter()
            .map(|op| {
                let state_clone = state.clone();
                tokio::spawn(async move { op(state_clone).await })
            })
            .collect();

        futures::future::join_all(handles).await;

        let final_state = state.lock().unwrap();
        $validator(&*final_state);
    };
}

/// Get current memory usage in bytes (platform-specific)
#[cfg(target_os = "linux")]
pub fn get_memory_usage() -> u64 {
    use std::fs;
    let contents = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in contents.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                return parts[1].parse::<u64>().unwrap_or(0) * 1024; // Convert KB to bytes
            }
        }
    }
    0
}

#[cfg(target_os = "macos")]
#[must_use]
pub fn get_memory_usage() -> u64 {
    // Simplified implementation for macOS
    // In a real implementation, you'd use mach APIs
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|output| {
            String::from_utf8(output.stdout)
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .map(|kb| kb * 1024) // Convert KB to bytes
        })
        .unwrap_or(0)
}

#[cfg(target_family = "wasm")]
pub fn get_memory_usage() -> u64 {
    // WASM doesn't have direct memory introspection
    js_sys::WebAssembly::Memory::from(wasm_bindgen::memory())
        .buffer()
        .byte_length() as u64
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_family = "wasm")))]
pub fn get_memory_usage() -> u64 {
    0 // Fallback for other platforms
}

#[cfg(test)]
mod tests {
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_assert_async_timeout_success() {
        let result = assert_async_timeout!(100, async {
            sleep(Duration::from_millis(10)).await;
            42
        });
        assert_eq!(result, 42);
    }

    #[tokio::test]
    #[should_panic(expected = "Operation timed out")]
    async fn test_assert_async_timeout_failure() {
        assert_async_timeout!(10, async {
            sleep(Duration::from_millis(100)).await;
            42
        });
    }
}
