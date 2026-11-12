use axum::{
    Router,
    http::{HeaderMap, StatusCode, header::AUTHORIZATION},
    routing::get,
};
use axum_test::TestServer;
use purr_common::hugging_face_token::HuggingFaceToken;
use purr_core::WhisperError;
use purr_core::error::{ConfigurationError, ModelIntegrityError};
use purr_core::model::download::download_verified;
use purr_core::model::pinned::PinnedFile;
use purr_core::model::registry::ModelSpec;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tempfile::TempDir;

mod common;
use common::{digest_of, fixture_bytes, fixture_registry};

fn fixture_spec(bytes: &[u8]) -> Result<ModelSpec, Box<dyn Error>> {
    Ok(fixture_registry(bytes)?.get("fixture")?.clone())
}

fn served(spec: &ModelSpec, url: String) -> PinnedFile {
    PinnedFile {
        url,
        ..spec.file.clone()
    }
}

fn serve(content: Vec<u8>) -> TestServer {
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
    TestServer::builder().http_transport().build(router)
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
    let server = serve(content.clone());
    let destination = Destination::new()?;

    let mut reports = Vec::new();
    download_verified(
        &served(&spec, url(&server, "/ok")?),
        None,
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
    let server = serve(content);
    let destination = Destination::new()?;

    let result = download_verified(
        &served(&spec, url(&server, "/tampered")?),
        None,
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
            assert_eq!(expected, spec.file.sha256);
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
    let server = serve(content.clone());
    let destination = Destination::new()?;

    let result = download_verified(
        &served(&spec, url(&server, "/short")?),
        None,
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
    let server = serve(content.clone());
    let destination = Destination::new()?;

    let result = download_verified(
        &served(&spec, url(&server, "/long")?),
        None,
        &destination.path,
        |_, _| {},
    )
    .await;

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
    let server = serve(content);
    let destination = Destination::new()?;

    let result = download_verified(
        &served(&spec, url(&server, "/missing")?),
        None,
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

#[tokio::test]
async fn the_token_is_not_sent_to_a_host_other_than_hugging_face() -> Result<(), Box<dyn Error>> {
    let content = fixture_bytes();
    let spec = fixture_spec(&content)?;
    let authorized = Arc::new(AtomicUsize::new(0));
    let router = Router::new().route(
        "/ok",
        get({
            let authorized = Arc::clone(&authorized);
            let content = content.clone();
            move |headers: HeaderMap| async move {
                if headers.contains_key(AUTHORIZATION) {
                    authorized.fetch_add(1, Ordering::SeqCst);
                }
                content
            }
        }),
    );
    let server = TestServer::builder().http_transport().build(router);
    let destination = Destination::new()?;
    let token = HuggingFaceToken::new("hf_dummy");

    download_verified(
        &served(&spec, url(&server, "/ok")?),
        token.as_ref(),
        &destination.path,
        |_, _| {},
    )
    .await?;

    assert_eq!(std::fs::read(&destination.path)?, content);
    assert_eq!(authorized.load(Ordering::SeqCst), 0);
    Ok(())
}
