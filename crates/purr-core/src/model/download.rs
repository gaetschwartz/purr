//! Checksum-verified model download.

use crate::error::{ConfigurationError, ModelIntegrityError, Result, WhisperError};
use crate::model::digest::Sha256Hasher;
use crate::model::manifest::Source;
use crate::model::registry::ModelSpec;
use futures_util::StreamExt as _;
use purr_common::hugging_face_token::HuggingFaceToken;
use std::path::Path;
use tokio::fs;
use tokio::io::AsyncWriteExt as _;
use tracing::warn;
use url::Url;

/// The token to send with a request to `url`: none unless `url` is on the Hugging Face origin.
///
/// A redirect to another host (the CDN) loses it, as `reqwest` drops `Authorization` there.
#[must_use]
pub fn token_for<'a>(
    url: &str,
    token: Option<&'a HuggingFaceToken>,
) -> Option<&'a HuggingFaceToken> {
    let token = token?;
    let hugging_face = Url::parse(Source::ORIGIN).ok()?;
    let target = Url::parse(url).ok()?;
    (target.origin() == hugging_face.origin()).then_some(token)
}

/// Download `url` to `destination`, publishing the file only if it matches
/// the size and SHA-256 recorded in `spec`.
///
/// With a `token`, a request to Hugging Face is authenticated with it.
/// The body is written to a temporary sibling file that is removed on any failure.
pub async fn download_verified<F>(
    spec: &ModelSpec,
    url: &str,
    token: Option<&HuggingFaceToken>,
    destination: &Path,
    mut progress: F,
) -> Result<()>
where
    F: FnMut(u64, Option<u64>),
{
    let mut request = crate::http::client_builder()?
        .build()
        .map_err(|source| WhisperError::from(ConfigurationError::DownloadFailed { source }))?
        .get(url);
    if let Some(token) = token_for(url, token) {
        request = request.bearer_auth(token.expose());
    }
    let response = request
        .send()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn token() -> Option<HuggingFaceToken> {
        HuggingFaceToken::new("hf_dummy")
    }

    fn sent_to(url: &str) -> bool {
        token_for(url, token().as_ref()).is_some()
    }

    #[test]
    fn requests_to_hugging_face_carry_the_token() {
        assert!(sent_to(
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/abc/ggml-base.bin"
        ));
        assert!(sent_to(
            "https://huggingface.co:443/owner/repo/resolve/abc/file"
        ));
        assert!(sent_to("https://huggingface.co"));
    }

    #[test]
    fn requests_to_any_other_host_do_not() {
        for url in [
            "http://huggingface.co/owner/repo",
            "https://huggingface.co.evil.example/owner/repo",
            "https://evil.example/?huggingface.co",
            "https://evil.example/huggingface.co",
            "https://huggingface.co@evil.example/",
            "https://hf.co/owner/repo",
            "https://us.aws.cdn.hf.co/repos/aa/bb/file",
            "https://cas-bridge.xethub.hf.co/xet-bridge-us/file",
            "https://cdn-lfs.huggingface.co/repos/file",
            "https://huggingface.co:8443/owner/repo",
            "http://127.0.0.1:8080/ok",
            "not a url",
        ] {
            assert_eq!(token_for(url, token().as_ref()), None, "{url}");
        }
    }

    #[test]
    fn no_token_stays_no_token() {
        assert_eq!(token_for("https://huggingface.co/owner/repo", None), None);
    }
}
