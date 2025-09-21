//! Shared test utilities and fixtures for the Purr workspace
//!
//! This crate provides:
//! - Common test fixtures and data generators
//! - Mock implementations for external dependencies
//! - WebGPU testing utilities
//! - Assertion macros for common patterns
//! - Property-based testing helpers

pub mod assertions;
pub mod fixtures;
pub mod generators;
pub mod mocks;

#[cfg(target_arch = "wasm32")]
pub mod webgpu;

#[cfg(feature = "benchmarks")]
pub mod benchmarks;

// Re-export commonly used testing dependencies
pub use pretty_assertions::{assert_eq as pretty_assert_eq, assert_ne as pretty_assert_ne};
pub use proptest::prelude::*;
pub use rstest::*;
pub use tempfile::TempDir;
pub use tokio_test;

// Re-export fake data generation
pub use fake::{Dummy, Fake, Faker};

/// Common test result type
pub type TestResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Create a temporary directory for tests
#[must_use]
pub fn temp_dir() -> TempDir {
    tempfile::tempdir().expect("Failed to create temporary directory")
}

/// Create a temporary file for tests
#[must_use]
pub fn temp_file() -> tempfile::NamedTempFile {
    tempfile::NamedTempFile::new().expect("Failed to create temporary file")
}

/// Initialize tracing for tests
pub fn init_test_tracing() {
    use tracing_subscriber::filter::EnvFilter;

    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::DEBUG.into()))
        .with_test_writer()
        .try_init();
}

/// Async test runner with timeout
pub async fn run_with_timeout<F, T>(timeout_ms: u64, test_fn: F) -> TestResult<T>
where
    F: std::future::Future<Output = TestResult<T>>,
{
    tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), test_fn)
        .await
        .map_err(|_| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "Test timed out",
            )) as Box<dyn std::error::Error + Send + Sync>
        })?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_temp_dir_creation() {
        let dir = temp_dir();
        assert!(dir.path().exists());
    }

    #[tokio::test]
    async fn test_timeout_runner() {
        let result = run_with_timeout(100, async {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            Ok(42)
        })
        .await;

        std::assert_eq!(result.unwrap(), 42);
    }
}
