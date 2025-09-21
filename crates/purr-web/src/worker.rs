//! WebGPU transcription using @huggingface/transformers
//! Direct integration with worker.js

use crate::error::{WebError, WebResult, WorkerError};
use crate::model::WebModelManager;
use crate::transcription::AudioMetadata;
use purr_common::platform::{TranscriptionRequest, TranscriptionStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex, RwLock};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::Stream;
use uuid::Uuid;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::{MessageEvent, Worker};

/// Transcription configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionConfig {
    pub model_name: String,
    pub language: Option<String>,
    pub translate: bool,
    pub quantized: bool,
    pub chunk_size: usize,
    pub temperature: f32,
    pub beam_size: usize,
    pub best_of: usize,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            model_name: "onnx-community/whisper-base".to_string(),
            language: None,
            translate: false,
            quantized: true,
            chunk_size: 30 * 16000, // 30 seconds at 16kHz
            temperature: 0.0,
            beam_size: 1,
            best_of: 1,
        }
    }
}

/// Messages sent to worker.js
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum WorkerMessage {
    Initialize {
        session_id: String,
        config: TranscriptionConfig,
    },
    StartTranscription {
        session_id: String,
        request_id: String,
        audio_data: Vec<u8>,
        language: Option<String>,
        translate: bool,
    },
    CancelTranscription {
        session_id: String,
        request_id: String,
    },
    ParseAudioMetadata {
        data: Vec<u8>,
        format: String,
    },
}

/// Messages received from worker.js
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum WorkerResponse {
    WorkerReady {
        session_id: String,
    },
    TranscriptionProgress {
        session_id: String,
        request_id: String,
        status: TranscriptionProgressStatus,
    },
    WorkerError {
        session_id: String,
        error: String,
    },
    AudioMetadataResponse {
        metadata: AudioMetadata,
    },
}

/// Transcription progress from worker.js
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum TranscriptionProgressStatus {
    Starting,
    ProcessingAudio,
    LoadingModel {
        progress: f64,
        message: String,
    },
    InProgress {
        chunk_index: usize,
        text: String,
        start_time: f64,
        end_time: f64,
    },
    Completed {
        processing_time: f64,
        audio_duration: f64,
        word_count: usize,
    },
    Error {
        message: String,
    },
}

/// Transcription worker
pub struct TranscriptionWorker {
    #[allow(dead_code)]
    model_manager: Arc<WebModelManager>,
    active_sessions: RwLock<HashMap<String, SessionData>>,
    worker_command_sender: Mutex<Option<mpsc::UnboundedSender<WorkerCommand>>>,
    ready_callbacks: Arc<RwLock<HashMap<String, oneshot::Sender<()>>>>,
}

#[derive(Debug)]
enum WorkerCommand {
    SendMessage(WorkerMessage),
    #[allow(dead_code)]
    InitializeWorker,
}

#[derive(Debug)]
struct SessionData {
    #[allow(dead_code)]
    config: TranscriptionConfig,
    active_requests: HashMap<String, mpsc::UnboundedSender<TranscriptionStatus>>,
}

impl TranscriptionWorker {
    /// Create worker instance
    #[must_use]
    pub fn new(model_manager: Arc<WebModelManager>) -> Self {
        Self {
            model_manager,
            active_sessions: RwLock::new(HashMap::new()),
            worker_command_sender: Mutex::new(None),
            ready_callbacks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create transcription session
    pub async fn create_session(&self, config: TranscriptionConfig) -> WebResult<String> {
        let session_id = Uuid::new_v4().to_string();

        // Initialize worker
        self.initialize_worker_if_needed().await?;

        // Send initialization to worker.js
        let init_message = WorkerMessage::Initialize {
            session_id: session_id.clone(),
            config: config.clone(),
        };

        self.send_to_worker(init_message).await?;

        // Wait for WorkerReady response
        self.wait_for_worker_ready(&session_id).await?;

        // Store session
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.insert(
                session_id.clone(),
                SessionData {
                    config,
                    active_requests: HashMap::new(),
                },
            );
        }

        tracing::info!("Created transcription session: {}", session_id);
        Ok(session_id)
    }

    /// Start transcription using WebGPU worker
    pub async fn transcribe(
        &self,
        session_id: &str,
        request: TranscriptionRequest,
    ) -> WebResult<impl Stream<Item = TranscriptionStatus>> {
        let request_id = Uuid::new_v4().to_string();
        let (tx, rx) = mpsc::unbounded_channel();

        // Add to active requests
        {
            let mut sessions = self.active_sessions.write().await;
            let session = sessions
                .get_mut(session_id)
                .ok_or_else(|| WebError::session_not_found(session_id))?;
            session
                .active_requests
                .insert(request_id.clone(), tx.clone());
        }

        // Send transcription request to worker.js
        let transcription_message = WorkerMessage::StartTranscription {
            session_id: session_id.to_string(),
            request_id: request_id.clone(),
            audio_data: request.file_data.to_vec(),
            language: request.language.clone(),
            translate: request.translate,
        };

        self.send_to_worker(transcription_message).await?;

        // Return stream
        Ok(UnboundedReceiverStream::new(rx))
    }

    /// Close transcription session
    pub async fn close_session(&self, session_id: &str) -> WebResult<()> {
        let mut sessions = self.active_sessions.write().await;
        sessions.remove(session_id);
        tracing::info!("Closed transcription session: {}", session_id);
        Ok(())
    }

    /// Initialize worker.js if needed
    async fn initialize_worker_if_needed(&self) -> WebResult<()> {
        let mut sender_guard = self.worker_command_sender.lock().await;

        if sender_guard.is_none() {
            let (command_tx, mut command_rx) = mpsc::unbounded_channel::<WorkerCommand>();

            // Spawn a local task to handle worker operations on the main thread
            let sessions = Arc::new(RwLock::new(HashMap::new()));
            let ready_callbacks_for_spawn = self.ready_callbacks.clone();
            spawn_local(async move {
                // Create web worker pointing to our worker.js
                // Use relative path from the web app's perspective
                let worker = match Worker::new("./worker.js") {
                    Ok(w) => w,
                    Err(e) => {
                        tracing::error!("Failed to create worker: {:?}", e);
                        return;
                    }
                };

                // Set up message handler
                let sessions_for_handler = sessions.clone();
                let ready_callbacks_for_handler = ready_callbacks_for_spawn.clone();
                let closure = Closure::wrap(Box::new(move |event: MessageEvent| {
                    let sessions = sessions_for_handler.clone();
                    let ready_callbacks = ready_callbacks_for_handler.clone();
                    spawn_local(async move {
                        Self::handle_worker_message(sessions, ready_callbacks, event).await;
                    });
                }) as Box<dyn FnMut(MessageEvent)>);

                worker.set_onmessage(Some(closure.as_ref().unchecked_ref()));
                closure.forget(); // Keep closure alive

                // Set up error handler
                let error_closure = Closure::wrap(Box::new(move |event: web_sys::ErrorEvent| {
                    web_sys::console::error_1(&format!("Worker error: {}", event.message()).into());
                })
                    as Box<dyn FnMut(web_sys::ErrorEvent)>);

                worker.set_onerror(Some(error_closure.as_ref().unchecked_ref()));
                error_closure.forget();

                tracing::info!("Initialized WebGPU worker");

                // Handle commands
                while let Some(command) = command_rx.recv().await {
                    match command {
                        WorkerCommand::SendMessage(message) => {
                            if let Ok(js_value) = serde_wasm_bindgen::to_value(&message) {
                                if let Err(e) = worker.post_message(&js_value) {
                                    tracing::error!("Failed to send message to worker: {:?}", e);
                                }
                            }
                        }
                        WorkerCommand::InitializeWorker => {
                            // Worker already initialized
                        }
                    }
                }
            });

            *sender_guard = Some(command_tx);
        }

        Ok(())
    }

    /// Send message to worker.js
    async fn send_to_worker(&self, message: WorkerMessage) -> WebResult<()> {
        let sender_guard = self.worker_command_sender.lock().await;
        let sender = sender_guard.as_ref().ok_or(WorkerError::NotInitialized)?;

        sender
            .send(WorkerCommand::SendMessage(message))
            .map_err(|_| WorkerError::CommandSendFailed)?;

        Ok(())
    }

    /// Wait for `WorkerReady` from worker.js
    async fn wait_for_worker_ready(&self, session_id: &str) -> WebResult<()> {
        let (ready_tx, ready_rx) = oneshot::channel();

        // Store the ready callback for this session
        {
            let mut ready_callbacks = self.ready_callbacks.write().await;
            ready_callbacks.insert(session_id.to_string(), ready_tx);
        }

        // Wait for actual WorkerReady message from worker.js with timeout
        tokio::time::timeout(std::time::Duration::from_secs(10), ready_rx)
            .await
            .map_err(|_| WorkerError::ReadyTimeout)?
            .map_err(|_| WorkerError::ReadyChannelClosed)?;

        tracing::info!("Worker ready for session: {}", session_id);
        Ok(())
    }

    /// Handle messages from worker.js
    async fn handle_worker_message(
        sessions: Arc<RwLock<HashMap<String, SessionData>>>,
        ready_callbacks: Arc<RwLock<HashMap<String, oneshot::Sender<()>>>>,
        event: MessageEvent,
    ) {
        if let Some(response_text) = event.data().as_string() {
            if let Ok(response) = serde_json::from_str::<WorkerResponse>(&response_text) {
                match response {
                    WorkerResponse::WorkerReady { session_id } => {
                        tracing::info!("Worker ready for session: {}", session_id);

                        // Signal the waiting thread that worker is ready
                        let mut callbacks = ready_callbacks.write().await;
                        if let Some(callback) = callbacks.remove(&session_id) {
                            let _ = callback.send(());
                        }
                    }
                    WorkerResponse::TranscriptionProgress {
                        session_id,
                        request_id,
                        status,
                    } => {
                        let sessions_guard = sessions.read().await;
                        if let Some(session) = sessions_guard.get(&session_id) {
                            if let Some(sender) = session.active_requests.get(&request_id) {
                                let transcription_status = Self::convert_progress_status(status);
                                let _ = sender.send(transcription_status);
                            }
                        }
                    }
                    WorkerResponse::WorkerError { session_id, error } => {
                        tracing::error!("Worker error for session {}: {}", session_id, error);
                        let sessions_guard = sessions.read().await;
                        if let Some(session) = sessions_guard.get(&session_id) {
                            for sender in session.active_requests.values() {
                                let _ = sender.send(TranscriptionStatus::Error {
                                    context: "Worker error".to_string(),
                                    error_message: error.clone(),
                                });
                            }
                        }
                    }
                    WorkerResponse::AudioMetadataResponse { metadata } => {
                        // This is handled here for future extensibility
                        tracing::info!("Received audio metadata response: {:?}", metadata);
                    }
                }
            }
        }
    }

    /// Convert worker progress to `TranscriptionStatus`
    fn convert_progress_status(status: TranscriptionProgressStatus) -> TranscriptionStatus {
        match status {
            TranscriptionProgressStatus::Starting => TranscriptionStatus::Starting,
            TranscriptionProgressStatus::ProcessingAudio => TranscriptionStatus::ProcessingAudio,
            TranscriptionProgressStatus::LoadingModel { .. } => {
                // For now, map to ProcessingAudio since TranscriptionStatus doesn't have LoadingModel
                TranscriptionStatus::ProcessingAudio
            }
            TranscriptionProgressStatus::InProgress {
                chunk_index,
                text,
                start_time,
                end_time,
            } => TranscriptionStatus::InProgress {
                chunk_index,
                text,
                start_time,
                end_time,
            },
            TranscriptionProgressStatus::Completed {
                processing_time,
                audio_duration,
                word_count,
            } => TranscriptionStatus::Completed {
                processing_time,
                audio_duration: audio_duration as f32,
                word_count,
            },
            TranscriptionProgressStatus::Error { message } => TranscriptionStatus::Error {
                context: "Progress error".to_string(),
                error_message: message,
            },
        }
    }
}

// Worker session status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatus {
    pub id: String,
    pub config: TranscriptionConfig,
    pub is_ready: bool,
    pub active_requests: usize,
}
