use axum::{Router, http::StatusCode, routing::get};
use axum_test::TestServer;
use purr_core::WhisperError;
use purr_core::error::{ConfigurationError, ModelIntegrityError};
use purr_core::model::digest::{Sha256Digest, Sha256Hasher};
use purr_core::model::download::download_verified;
use purr_core::model::registry::{ModelSpec, Registry};
use std::error::Error;
use std::path::PathBuf;
use tempfile::TempDir;

const REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";

fn fixture_bytes() -> Vec<u8> {
    (0..300_000u32).map(|i| (i % 251) as u8).collect()
}

fn digest_of(bytes: &[u8]) -> Sha256Digest {
    let mut hasher = Sha256Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

fn fixture_spec(bytes: &[u8]) -> Result<ModelSpec, Box<dyn Error>> {
    let json = serde_json::json!({
        "schema": 1,
        "preferred": ["fixture"],
        "sources": [{"id": "main", "repo": "owner/repo", "revision": REVISION}],
        "models": [{
            "id": "fixture", "source": "main", "filename": "ggml-fixture.bin",
            "family": "tiny", "language": "multilingual", "quantization": "fp16",
            "features": [], "size_bytes": bytes.len(), "sha256": digest_of(bytes).to_string(),
            "description": "Fixture"
        }],
    });
    let registry = Registry::from_json(&json.to_string())?;
    Ok(registry.get("fixture")?.clone())
}

async fn serve(content: Vec<u8>) -> Result<TestServer, Box<dyn Error>> {
    let mut tampered = content.clone();
    if let Some(first) = tampered.first_mut() {
        *first ^= 0xFF;
    }
    let mut short = content.clone();
    short.pop();
    let mut long = content.clone();
    long.extend_from_slice(&[0u8; 1024]);

    let router = Router::new()
        .route("/ok", get(move || async move { content }))
        .route("/tampered", get(move || async move { tampered }))
        .route("/short", get(move || async move { short }))
        .route("/long", get(move || async move { long }))
        .route("/missing", get(|| async { StatusCode::NOT_FOUND }));
    Ok(TestServer::builder().http_transport().build(router))
}

fn url(server: &TestServer, path: &str) -> Result<String, Box<dyn Error>> {
    let base = server
        .server_address()
        .ok_or("test server has no network address")?;
    Ok(format!("{}{path}", base.as_str().trim_end_matches('/')))
}

struct Destination {
    _dir: TempDir,
    path: PathBuf,
}

impl Destination {
    fn new() -> Result<Self, Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("ggml-fixture.bin");
        Ok(Self { _dir: dir, path })
    }

    fn temp_path(&self) -> PathBuf {
        self.path.with_extension("tmp")
    }
}

#[tokio::test]
async fn verified_download_publishes_the_file_and_reports_progress() -> Result<(), Box<dyn Error>> {
    let content = fixture_bytes();
    let spec = fixture_spec(&content)?;
    let server = serve(content.clone()).await?;
    let destination = Destination::new()?;

    let mut reports = Vec::new();
    download_verified(
        &spec,
        &url(&server, "/ok")?,
        &destination.path,
        |downloaded, total| reports.push((downloaded, total)),
    )
    .await?;

    assert_eq!(std::fs::read(&destination.path)?, content);
    assert!(!destination.temp_path().exists());
    assert_eq!(
        reports.last().copied(),
        Some((content.len() as u64, Some(content.len() as u64)))
    );
    assert!(reports.windows(2).all(|pair| pair[0].0 < pair[1].0));
    Ok(())
}

#[tokio::test]
async fn checksum_mismatch_is_rejected_and_cleaned_up() -> Result<(), Box<dyn Error>> {
    let content = fixture_bytes();
    let spec = fixture_spec(&content)?;
    let mut tampered = content.clone();
    if let Some(first) = tampered.first_mut() {
        *first ^= 0xFF;
    }
    let server = serve(content).await?;
    let destination = Destination::new()?;

    let result = download_verified(
        &spec,
        &url(&server, "/tampered")?,
        &destination.path,
        |_, _| {},
    )
    .await;

    match result {
        Err(WhisperError::Configuration {
            source:
                ConfigurationError::ModelIntegrity(ModelIntegrityError::ChecksumMismatch {
                    model_id,
                    expected,
                    actual,
                }),
        }) => {
            assert_eq!(model_id, "fixture");
            assert_eq!(expected, spec.sha256);
            assert_eq!(actual, digest_of(&tampered));
        }
        other => return Err(format!("expected a checksum mismatch, got {other:?}").into()),
    }
    assert!(!destination.path.exists());
    assert!(!destination.temp_path().exists());
    Ok(())
}

#[tokio::test]
async fn truncated_body_is_rejected_and_cleaned_up() -> Result<(), Box<dyn Error>> {
    let content = fixture_bytes();
    let spec = fixture_spec(&content)?;
    let server = serve(content.clone()).await?;
    let destination = Destination::new()?;

    let result = download_verified(
        &spec,
        &url(&server, "/short")?,
        &destination.path,
        |_, _| {},
    )
    .await;

    match result {
        Err(WhisperError::Configuration {
            source:
                ConfigurationError::ModelIntegrity(ModelIntegrityError::SizeMismatch {
                    model_id,
                    expected,
                    actual,
                }),
        }) => {
            assert_eq!(model_id, "fixture");
            assert_eq!(expected, content.len() as u64);
            assert_eq!(actual, content.len() as u64 - 1);
        }
        other => return Err(format!("expected a size mismatch, got {other:?}").into()),
    }
    assert!(!destination.path.exists());
    assert!(!destination.temp_path().exists());
    Ok(())
}

#[tokio::test]
async fn oversized_body_is_aborted_and_cleaned_up() -> Result<(), Box<dyn Error>> {
    let content = fixture_bytes();
    let spec = fixture_spec(&content)?;
    let server = serve(content.clone()).await?;
    let destination = Destination::new()?;

    let result =
        download_verified(&spec, &url(&server, "/long")?, &destination.path, |_, _| {}).await;

    match result {
        Err(WhisperError::Configuration {
            source:
                ConfigurationError::ModelIntegrity(ModelIntegrityError::SizeMismatch {
                    expected,
                    actual,
                    ..
                }),
        }) => {
            assert_eq!(expected, content.len() as u64);
            assert!(actual > expected);
        }
        other => return Err(format!("expected a size mismatch, got {other:?}").into()),
    }
    assert!(!destination.path.exists());
    assert!(!destination.temp_path().exists());
    Ok(())
}

#[tokio::test]
async fn http_error_status_leaves_no_files() -> Result<(), Box<dyn Error>> {
    let content = fixture_bytes();
    let spec = fixture_spec(&content)?;
    let server = serve(content).await?;
    let destination = Destination::new()?;

    let result = download_verified(
        &spec,
        &url(&server, "/missing")?,
        &destination.path,
        |_, _| {},
    )
    .await;

    assert!(matches!(
        result,
        Err(WhisperError::Configuration {
            source: ConfigurationError::DownloadHttpError {
                status_code: 404,
                ..
            }
        })
    ));
    assert!(!destination.path.exists());
    assert!(!destination.temp_path().exists());
    Ok(())
}
