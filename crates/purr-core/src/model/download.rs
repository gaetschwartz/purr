//! Checksum-verified model download.

use crate::error::{ConfigurationError, ModelIntegrityError, Result, WhisperError};
use crate::model::digest::Sha256Hasher;
use crate::model::registry::ModelSpec;
use futures_util::StreamExt as _;
use std::path::Path;
use tokio::fs;
use tokio::io::AsyncWriteExt as _;
use tracing::warn;

/// Download `url` to `destination`, publishing the file only if it matches
/// the size and SHA-256 recorded in `spec`.
///
/// The body is written to a temporary sibling file that is removed on any failure.
pub async fn download_verified<F>(
    spec: &ModelSpec,
    url: &str,
    destination: &Path,
    mut progress: F,
) -> Result<()>
where
    F: FnMut(u64, Option<u64>),
{
    let response = reqwest::get(url)
        .await
        .map_err(|source| WhisperError::from(ConfigurationError::DownloadFailed { source }))?;

    if !response.status().is_success() {
        return Err(WhisperError::from(ConfigurationError::DownloadHttpError {
            model_name: spec.id.clone(),
            status_code: response.status().as_u16(),
        }));
    }

    let total = response.content_length().unwrap_or(spec.size_bytes);
    let temp_path = destination.with_extension("tmp");

    let outcome: Result<()> = async {
        stream_verified(spec, response, &temp_path, |downloaded| {
            progress(downloaded, Some(total));
        })
        .await?;
        fs::rename(&temp_path, destination).await?;
        Ok(())
    }
    .await;

    if outcome.is_err() {
        discard(&temp_path).await;
    }
    outcome
}

async fn stream_verified<F>(
    spec: &ModelSpec,
    response: reqwest::Response,
    temp_path: &Path,
    mut on_progress: F,
) -> Result<()>
where
    F: FnMut(u64),
{
    let mut file = fs::File::create(temp_path).await?;
    let mut hasher = Sha256Hasher::new();
    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk
            .map_err(|source| ConfigurationError::DownloadStreamError { source })
            .map_err(WhisperError::from)?;

        downloaded += chunk.len() as u64;
        if downloaded > spec.size_bytes {
            return Err(size_mismatch(spec, downloaded));
        }

        hasher.update(&chunk);
        file.write_all(&chunk).await?;
        on_progress(downloaded);
    }

    file.flush().await?;
    drop(file);

    if downloaded != spec.size_bytes {
        return Err(size_mismatch(spec, downloaded));
    }

    let actual = hasher.finalize();
    if actual != spec.sha256 {
        return Err(WhisperError::from(ModelIntegrityError::ChecksumMismatch {
            model_id: spec.id.clone(),
            expected: spec.sha256,
            actual,
        }));
    }
    Ok(())
}

fn size_mismatch(spec: &ModelSpec, actual: u64) -> WhisperError {
    WhisperError::from(ModelIntegrityError::SizeMismatch {
        model_id: spec.id.clone(),
        expected: spec.size_bytes,
        actual,
    })
}

async fn discard(path: &Path) {
    if let Err(err) = fs::remove_file(path).await
        && err.kind() != std::io::ErrorKind::NotFound
    {
        warn!("Failed to remove temporary file {}: {err}", path.display());
    }
}
