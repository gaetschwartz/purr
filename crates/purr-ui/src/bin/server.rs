/// Separate HTTP server binary for native backend
/// This handles file upload and transcription via REST APIs
use axum::{
    extract::{Multipart, Query},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use uuid::Uuid;

#[derive(Serialize, Deserialize)]
struct UploadResponse {
    file_id: String,
    message: String,
}

#[derive(Serialize, Deserialize)]
struct TranscriptionRequest {
    file_id: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    translate: bool,
}

#[derive(Serialize, Deserialize)]
struct TranscriptionResponse {
    text: String,
    processing_time: f64,
    word_count: usize,
}

/// Handle file upload
async fn upload_file(mut multipart: Multipart) -> Result<Json<UploadResponse>, StatusCode> {
    while let Some(field) = multipart.next_field().await.map_err(|_| StatusCode::BAD_REQUEST)? {
        if field.name() == Some("file") || field.name() == Some("audio") {
            let file_name = field.file_name().unwrap_or("unknown").to_string();
            let data = field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?;

            // Generate a unique file ID
            let file_id = Uuid::new_v4().to_string();

            // Create upload directory
            let upload_dir = std::env::temp_dir().join("purr-uploads");
            tokio::fs::create_dir_all(&upload_dir)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            // Save file
            let file_path = upload_dir.join(&file_id);
            tokio::fs::write(&file_path, &data)
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            info!("Uploaded file: {} -> {}", file_name, file_id);

            return Ok(Json(UploadResponse {
                file_id,
                message: format!("File '{}' uploaded successfully", file_name),
            }));
        }
    }

    Err(StatusCode::BAD_REQUEST)
}

/// Handle transcription request
async fn transcribe_file(
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<TranscriptionResponse>, StatusCode> {
    let file_id = params
        .get("file_id")
        .ok_or(StatusCode::BAD_REQUEST)?
        .clone();

    info!("Starting real transcription for file: {}", file_id);

    // Get the uploaded file path
    let upload_dir = std::env::temp_dir().join("purr-uploads");
    let file_path = upload_dir.join(&file_id);

    if !file_path.exists() {
        return Err(StatusCode::NOT_FOUND);
    }

    // Use purr-core for real transcription
    let start_time = std::time::Instant::now();
    match purr_core::transcribe_file_sync(file_path, None).await {
        Ok(result) => {
            let processing_time = start_time.elapsed().as_secs_f64();
            let word_count = result.text.split_whitespace().count();
            info!("Transcription completed in {:.2}s", processing_time);

            Ok(Json(TranscriptionResponse {
                text: result.text,
                processing_time,
                word_count,
            }))
        }
        Err(e) => {
            tracing::error!("Transcription failed: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Health check endpoint
async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "purr-backend",
        "version": "0.1.0"
    }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    info!("Starting Purr backend server on port 8080");

    // Build the router
    let app = Router::new()
        .route("/api/health", get(health_check))
        .route("/api/upload", post(upload_file))
        .route("/api/transcribe", get(transcribe_file))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    // Start the server
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    info!("Backend server listening on http://127.0.0.1:8080");

    axum::serve(listener, app).await?;

    Ok(())
}