//! Web platform implementation of the Platform trait

use crate::audio;
use crate::engine_choice::{self, GpuFailure};
use crate::error::{IntegrityError, WebError, WebResult};
use crate::history_store;
use crate::local::run_local;
use crate::model::WebModelManager;
use crate::model_cache;
use crate::model_info::model_info;
use crate::protocol::{WorkerEvent, WorkerModel, WorkerRequest};
use crate::status_mapper::{PreloadMapper, PreloadStep, TranscriptionMapper};
use crate::storage::WebStorage;
use crate::storage_space;
use crate::worker_client::WorkerClient;
use bytes::Bytes;
use futures::StreamExt as _;
use purr_common::accel::AccelPolicy;
use purr_common::accel::SystemReport;
use purr_common::adapter_name::{AdapterStrings, adapter_name};
use purr_common::audio_overview::{AudioOverview, peaks};
use purr_common::browser_engine::CpuReason;
use purr_common::history::{
    entry::{HistoryEntry, HistoryId, HistoryRecord},
    listing::{HistoryListing, Saved},
    retention::Limits,
};
use purr_common::hugging_face_token::HuggingFaceToken;
use purr_common::platform::{
    DeviceInfo, DeviceType, FileId, FileSource, ModelCheckSender, ModelFailureKind, ModelInfo,
    ModelIntegrity, ModelOperationProgress, ModelProgressStream, Platform, PlatformError,
    TranscriptionRequest, TranscriptionStatus, TranscriptionStream,
};
use purr_common::settings::Settings;
use purr_common::web_model::registry::WebModelSpec;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;
use wasm_bindgen::JsCast as _;
use wasm_bindgen_futures::{JsFuture, spawn_local};
use web_sys::{DomException, Storage};

pub type PlatformImpl = WasmPlatformImpl;

const SETTINGS_KEY: &str = "purr-settings";
const RETIRED_GPU_CHECKS_KEY: &str = "purr-gpu-checks";
const UPLOAD_RETENTION_MS: f64 = 24.0 * 60.0 * 60.0 * 1000.0;

/// Web platform implementation using WebAssembly and browser APIs
pub struct WasmPlatformImpl {
    storage: Arc<WebStorage>,
    model_manager: Arc<WebModelManager>,
    worker: Arc<WorkerClient>,
    initialized: RwLock<bool>,
    config_mutex: Mutex<()>,
    history_lock: Mutex<()>,
    decode_lock: Arc<Mutex<()>>,
}

impl WasmPlatformImpl {
    async fn ensure_initialized(&self) -> Result<(), PlatformError> {
        if *self.initialized.read().await {
            return Ok(());
        }

        let _guard = self.config_mutex.lock().await;
        if *self.initialized.read().await {
            return Ok(());
        }

        self.storage.initialize().await?;
        self.storage.purge_older_than(UPLOAD_RETENTION_MS).await?;
        Self::remove_retired_gpu_checks();

        tracing::info!("Platform initialized");
        *self.initialized.write().await = true;
        Ok(())
    }

    fn remove_retired_gpu_checks() {
        let removed = Self::get_local_storage().and_then(|storage| {
            storage.remove_item(RETIRED_GPU_CHECKS_KEY).map_err(|e| {
                PlatformError::settings_persistence(format!(
                    "Failed to remove item '{RETIRED_GPU_CHECKS_KEY}' from localStorage: {}",
                    crate::error::format_js_error(e)
                ))
            })
        });
        if let Err(error) = removed {
            tracing::warn!("The retired GPU checks stay in localStorage: {error}");
        }
    }

    fn get_local_storage() -> Result<Storage, PlatformError> {
        let window = web_sys::window()
            .ok_or_else(|| PlatformError::settings_persistence("No window available".to_owned()))?;
        window
            .local_storage()
            .map_err(|e| {
                PlatformError::settings_persistence(format!("Failed to access localStorage: {e:?}"))
            })?
            .ok_or_else(|| {
                PlatformError::settings_persistence("localStorage not available".to_owned())
            })
    }

    fn get_storage_value(key: &str) -> Result<Option<String>, PlatformError> {
        let storage = Self::get_local_storage()?;
        storage.get_item(key).map_err(|e| {
            PlatformError::settings_persistence(format!(
                "Failed to get item '{key}' from localStorage: {e:?}"
            ))
        })
    }

    fn set_storage_value(key: &str, value: &str) -> Result<(), PlatformError> {
        let storage = Self::get_local_storage()?;
        storage.set_item(key, value).map_err(|e| {
            let is_quota = e
                .dyn_ref::<DomException>()
                .is_some_and(|exception| exception.name() == "QuotaExceededError");
            let detail = crate::error::format_js_error(e);
            if is_quota {
                PlatformError::settings_persistence(format!(
                    "localStorage quota exceeded; clear browser storage for this site: {detail}"
                ))
            } else {
                PlatformError::settings_persistence(format!(
                    "Failed to set item '{key}' in localStorage: {detail}"
                ))
            }
        })
    }

    pub fn new() -> Result<Self, PlatformError> {
        let storage = Arc::new(WebStorage::new());
        let model_manager = Arc::new(WebModelManager::new()?);
        let worker = Arc::new(WorkerClient::spawn(crate::worker_url()));
        Ok(Self {
            storage,
            model_manager,
            worker,
            initialized: RwLock::new(false),
            config_mutex: Mutex::new(()),
            history_lock: Mutex::new(()),
            decode_lock: Arc::new(Mutex::new(())),
        })
    }
}

fn now_ms() -> f64 {
    js_sys::Date::now()
}

fn verification_failure_kind(error: &WebError) -> ModelFailureKind {
    if matches!(error, WebError::Integrity(_)) {
        ModelFailureKind::Integrity
    } else {
        ModelFailureKind::Other
    }
}

/// Why a freshly downloaded model failed its check: a file missing because the site ran out of
/// storage is reported as such, since the browser's cache drops writes over quota silently.
async fn downloaded_check_failure(
    spec: &WebModelSpec,
    error: &WebError,
) -> (String, ModelFailureKind) {
    if matches!(error, WebError::Integrity(IntegrityError::Missing { .. })) {
        match storage_space::estimate().await {
            Ok(space) if space.free() < spec.total_size_bytes() => {
                return (
                    format!(
                        "The browser has no room left for this model: it needs {:.1} MiB and this site has {:.1} MiB free. Remove other models or free up disk space, then try again.",
                        mebibytes(spec.total_size_bytes()),
                        mebibytes(space.free())
                    ),
                    ModelFailureKind::Other,
                );
            }
            Ok(_) => {}
            Err(estimate_error) => {
                tracing::warn!("Could not read how much storage is left: {estimate_error}");
            }
        }
        return (
            format!(
                "The download finished but the browser did not keep all of it ({error}). Try again."
            ),
            ModelFailureKind::Other,
        );
    }
    (error.to_string(), verification_failure_kind(error))
}

fn mebibytes(bytes: u64) -> f64 {
    bytes as f64 / 1_048_576.0
}

fn download_failure_kind() -> ModelFailureKind {
    let offline = web_sys::window().is_some_and(|window| !window.navigator().on_line());
    if offline {
        ModelFailureKind::Network
    } else {
        ModelFailureKind::Other
    }
}

fn send_status(
    tx: &mpsc::UnboundedSender<Result<TranscriptionStatus, PlatformError>>,
    status: TranscriptionStatus,
) -> bool {
    tx.send(Ok(status)).is_ok()
}

/// Everything one transcription needs besides the channel it reports on.
struct Transcription {
    decode_lock: Arc<Mutex<()>>,
    worker: Arc<WorkerClient>,
    spec: WebModelSpec,
    model_cached: bool,
    bytes: Bytes,
    language: Option<String>,
    translate: bool,
    hugging_face_token: Option<HuggingFaceToken>,
    policy: AccelPolicy,
}

async fn run_transcription(
    job: Transcription,
    tx: mpsc::UnboundedSender<Result<TranscriptionStatus, PlatformError>>,
) {
    let Transcription {
        decode_lock,
        worker,
        spec,
        model_cached,
        bytes,
        language,
        translate,
        hugging_face_token,
        policy,
    } = job;
    if !send_status(&tx, TranscriptionStatus::Starting)
        || !send_status(&tx, TranscriptionStatus::ProcessingAudio)
    {
        return;
    }

    let decoding = decode_lock.lock_owned().await;
    let decoded = audio::decode(bytes).await;
    drop(decoding);
    let decoded = match decoded {
        Ok(decoded) => decoded,
        Err(error) => {
            send_status(
                &tx,
                TranscriptionStatus::Error {
                    context: "Audio".to_owned(),
                    error_message: error.to_string(),
                },
            );
            return;
        }
    };
    tracing::info!(
        "Decoded {:.1}s of audio for model {}",
        decoded.duration_seconds(),
        spec.id
    );

    let mut mapper = TranscriptionMapper::new(&spec, model_cached);
    let mut choice = match engine_choice::choose(&spec, policy) {
        Ok(choice) => choice,
        Err(reason) => {
            send_status(&tx, cannot_run(reason));
            return;
        }
    };
    let samples = decoded.samples;
    loop {
        tracing::info!("Transcribing with {} on {:?}", spec.id, choice.engine);
        if !send_status(
            &tx,
            TranscriptionStatus::BrowserEngine {
                engine: choice.engine,
            },
        ) {
            return;
        }
        let request = WorkerRequest::Transcribe {
            request_id: worker.next_request_id(),
            model: WorkerModel::from(&spec),
            provider: choice.provider(),
            language: language.clone(),
            translate,
            english_only: spec.english_only(),
            hugging_face_token: hugging_face_token.clone(),
        };
        let mut events = match worker.submit(request, Some(samples.clone())) {
            Ok(events) => events,
            Err(error) => {
                send_status(
                    &tx,
                    TranscriptionStatus::InitFailed {
                        component: "Worker".to_owned(),
                        reason: error.to_string(),
                        error_details: None,
                    },
                );
                return;
            }
        };
        let fallback = loop {
            tokio::select! {
                () = tx.closed() => {
                    tracing::info!("Transcription stream dropped; cancelling");
                    return;
                }
                next = events.next() => {
                    let Some(event) = next else { return };
                    if let WorkerEvent::Failed { kind, message, .. } = &event {
                        match choice.after_failure(*kind) {
                            GpuFailure::MoveToCpu(cpu) => {
                                tracing::warn!("WebGPU failed, moving to the CPU: {message}");
                                worker.release_idle();
                                break cpu;
                            }
                            GpuFailure::CannotRun(reason) => {
                                tracing::warn!("WebGPU failed and {} cannot run on the CPU: {message}", spec.id);
                                worker.release_idle();
                                send_status(&tx, cannot_run(reason));
                                return;
                            }
                            GpuFailure::Unrelated => {}
                        }
                    }
                    if let Some(status) = mapper.map(event) && !send_status(&tx, status) {
                        return;
                    }
                }
            }
        };
        choice = fallback;
    }
}

/// The failure of a run whose model only runs on WebGPU when the CPU would have to take it.
fn cannot_run(reason: CpuReason) -> TranscriptionStatus {
    TranscriptionStatus::InitFailed {
        component: "WebGPU".to_owned(),
        reason: reason.gpu_only_refusal().to_owned(),
        error_details: Some("Choose a smaller model; it can run on the CPU.".to_owned()),
    }
}

fn load_request(
    worker: &WorkerClient,
    spec: &WebModelSpec,
    choice: &engine_choice::Choice,
    hugging_face_token: Option<&HuggingFaceToken>,
) -> WorkerRequest {
    WorkerRequest::LoadModel {
        request_id: worker.next_request_id(),
        model: WorkerModel::from(spec),
        provider: choice.provider(),
        hugging_face_token: hugging_face_token.cloned(),
    }
}

async fn run_preload(
    worker: Arc<WorkerClient>,
    spec: WebModelSpec,
    hugging_face_token: Option<HuggingFaceToken>,
    policy: AccelPolicy,
    tx: mpsc::UnboundedSender<Result<ModelOperationProgress, PlatformError>>,
) {
    let model_id = spec.id.clone();
    let fail = |operation: &str, error_message: String, kind: ModelFailureKind| {
        ModelOperationProgress::Failed {
            model_id: model_id.clone(),
            operation: operation.to_owned(),
            error_message,
            kind,
        }
    };
    if tx
        .send(Ok(ModelOperationProgress::Starting {
            model_id: model_id.clone(),
        }))
        .is_err()
    {
        return;
    }

    let verified_first = match model_cache::is_installed(&spec).await {
        Ok(true) => {
            if tx
                .send(Ok(ModelOperationProgress::Installing {
                    model_id: model_id.clone(),
                }))
                .is_err()
            {
                return;
            }
            if let Err(error) = model_cache::verify(&spec).await {
                if tx
                    .send(Ok(fail(
                        "verification",
                        error.to_string(),
                        verification_failure_kind(&error),
                    )))
                    .is_err()
                {
                    tracing::warn!("Verification of {model_id} failed after its listener left");
                }
                return;
            }
            true
        }
        Ok(false) => false,
        Err(error) => {
            if tx
                .send(Ok(fail(
                    "verification",
                    error.to_string(),
                    ModelFailureKind::Other,
                )))
                .is_err()
            {
                tracing::warn!("Checking {model_id} failed after its listener left");
            }
            return;
        }
    };

    let mut choice = match engine_choice::choose(&spec, policy) {
        Ok(choice) => choice,
        Err(reason) => {
            if tx
                .send(Ok(fail(
                    "download",
                    reason.gpu_only_refusal().to_owned(),
                    ModelFailureKind::Other,
                )))
                .is_err()
            {
                tracing::warn!("Preload of {model_id} failed after its listener left");
            }
            return;
        }
    };
    let mut events = match worker.submit(
        load_request(&worker, &spec, &choice, hugging_face_token.as_ref()),
        None,
    ) {
        Ok(events) => events,
        Err(error) => {
            let failure = fail("download", error.to_string(), ModelFailureKind::Other);
            if tx.send(Ok(failure)).is_err() {
                tracing::warn!("Preload of {model_id} failed after its listener left");
            }
            return;
        }
    };

    let mut mapper = PreloadMapper::new(&spec, now_ms());
    loop {
        tokio::select! {
            () = tx.closed() => {
                tracing::info!("Preload stream dropped; cancelling");
                return;
            }
            next = events.next() => {
                let Some(event) = next else { return };
                if let WorkerEvent::Failed { kind, message, .. } = &event {
                    match choice.after_failure(*kind) {
                        GpuFailure::MoveToCpu(cpu) => {
                            tracing::warn!("WebGPU could not load {model_id}, loading it for the CPU: {message}");
                            choice = cpu;
                            worker.release_idle();
                            match worker.submit(load_request(&worker, &spec, &choice, hugging_face_token.as_ref()), None) {
                                Ok(retry) => {
                                    events = retry;
                                    continue;
                                }
                                Err(error) => {
                                    if tx.send(Ok(fail("download", error.to_string(), ModelFailureKind::Other))).is_err() {
                                        tracing::warn!("Preload of {model_id} failed after its listener left");
                                    }
                                    return;
                                }
                            }
                        }
                        GpuFailure::CannotRun(reason) => {
                            worker.release_idle();
                            if tx.send(Ok(fail("download", reason.gpu_only_refusal().to_owned(), ModelFailureKind::Other))).is_err() {
                                tracing::warn!("Preload of {model_id} failed after its listener left");
                            }
                            return;
                        }
                        GpuFailure::Unrelated => {}
                    }
                }
                let outcome = match mapper.map(event, now_ms()) {
                    PreloadStep::Progress(progress) => Ok(progress),
                    PreloadStep::Ignored => continue,
                    PreloadStep::Failed(message) => {
                        worker.release_idle();
                        Ok(fail("download", message, download_failure_kind()))
                    }
                    PreloadStep::Loaded => {
                        if !verified_first
                            && tx
                                .send(Ok(ModelOperationProgress::Installing { model_id: model_id.clone() }))
                                .is_err()
                        {
                            return;
                        }
                        let verified = if verified_first {
                            Ok(())
                        } else {
                            model_cache::verify(&spec).await
                        };
                        match verified {
                            Ok(()) => Ok(ModelOperationProgress::Completed {
                                model_id: model_id.clone(),
                                local_path: PathBuf::from(model_cache::location(&spec)),
                            }),
                            Err(error) => {
                                worker.release_idle();
                                let (message, kind) = downloaded_check_failure(&spec, &error).await;
                                Ok(fail("verification", message, kind))
                            }
                        }
                    }
                };
                let finished = matches!(
                    outcome,
                    Ok(ModelOperationProgress::Completed { .. } | ModelOperationProgress::Failed { .. })
                );
                if tx.send(outcome).is_err() || finished {
                    return;
                }
            }
        }
    }
}

#[allow(
    clippy::unused_async_trait_impl,
    reason = "the trait methods are async"
)]
impl Platform for WasmPlatformImpl {
    fn install_logging_hooks(&self) {}

    async fn process_file(
        &self,
        file_data: Bytes,
        file_path: &Path,
    ) -> Result<FileId, PlatformError> {
        audio::validate_size(file_data.len()).map_err(WebError::from)?;
        self.ensure_initialized().await?;

        let file_name = file_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");
        let mime_type = match file_path.extension().and_then(|ext| ext.to_str()) {
            Some("mp3") => "audio/mpeg",
            Some("wav") => "audio/wav",
            Some("flac") => "audio/flac",
            Some("ogg" | "opus") => "audio/ogg",
            Some("m4a" | "mp4") => "audio/mp4",
            Some("aac") => "audio/aac",
            Some("webm") => "audio/webm",
            _ => "application/octet-stream",
        };

        Ok(self
            .storage
            .store_file(file_data, file_name, mime_type)
            .await?)
    }

    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionStream, PlatformError> {
        self.ensure_initialized().await?;

        let bytes = match &request.file {
            FileSource::Bytes(bytes) => bytes.clone(),
            FileSource::Path(path) => {
                return Err(PlatformError::unsupported(format!(
                    "transcribing the file path {} in the browser; upload the file instead",
                    path.display()
                )));
            }
            FileSource::Uploaded(file_id) => self
                .storage
                .get_file(file_id)
                .await?
                .ok_or_else(|| PlatformError::file_processing("File not found".to_owned()))?,
        };
        audio::validate_size(bytes.len()).map_err(WebError::from)?;

        let spec = self.model_manager.model(request.model.as_str())?.clone();
        let model_cached = self.model_manager.is_model_installed(&spec.id).await?;
        let language = request.language;

        tracing::info!(
            "Transcribing {} bytes with {} ({}), language {:?}",
            bytes.len(),
            spec.id,
            spec.repo,
            language
        );

        let (tx, rx) = mpsc::unbounded_channel();
        spawn_local(run_transcription(
            Transcription {
                decode_lock: Arc::clone(&self.decode_lock),
                worker: Arc::clone(&self.worker),
                spec,
                model_cached,
                bytes,
                language,
                translate: request.translate,
                hugging_face_token: request.hugging_face_token,
                policy: request.engine.accel,
            },
            tx,
        ));
        Ok(Box::pin(UnboundedReceiverStream::new(rx)))
    }

    async fn cleanup(&self, file_id: &str) -> Result<(), PlatformError> {
        self.ensure_initialized().await?;
        self.storage.delete_file(&FileId::from(file_id)).await?;
        Ok(())
    }

    fn max_upload_bytes(&self) -> Option<u64> {
        Some(audio::MAX_AUDIO_BYTES as u64)
    }

    async fn audio_overview(
        &self,
        file_id: &FileId,
        buckets: usize,
    ) -> Result<AudioOverview, PlatformError> {
        self.ensure_initialized().await?;
        let bytes = self
            .storage
            .get_file(file_id)
            .await?
            .ok_or_else(|| PlatformError::file_processing("File not found".to_owned()))?;
        let _one_decode_at_a_time = self.decode_lock.lock().await;
        let decoded = audio::decode(bytes).await?;
        Ok(AudioOverview {
            duration_seconds: decoded.duration_seconds(),
            peaks: peaks(&decoded.samples, buckets),
        })
    }

    async fn list_history(&self) -> Result<HistoryListing, PlatformError> {
        self.ensure_initialized().await?;
        let _one_at_a_time = self.history_lock.lock().await;
        Ok(history_store::list().await?)
    }

    async fn get_history_entry(&self, id: &HistoryId) -> Result<HistoryRecord, PlatformError> {
        self.ensure_initialized().await?;
        let _one_at_a_time = self.history_lock.lock().await;
        Ok(history_store::get(id).await?)
    }

    async fn save_history_entry(&self, entry: &HistoryEntry) -> Result<Saved, PlatformError> {
        self.ensure_initialized().await?;
        let _one_at_a_time = self.history_lock.lock().await;
        Ok(history_store::save(entry, Limits::STANDARD).await?)
    }

    async fn rename_history_entry(&self, id: &HistoryId, title: &str) -> Result<(), PlatformError> {
        self.ensure_initialized().await?;
        let _one_at_a_time = self.history_lock.lock().await;
        Ok(history_store::rename(id, title).await?)
    }

    async fn delete_history_entry(&self, id: &HistoryId) -> Result<(), PlatformError> {
        self.ensure_initialized().await?;
        let _one_at_a_time = self.history_lock.lock().await;
        Ok(history_store::delete(id).await?)
    }

    async fn clear_history(&self) -> Result<usize, PlatformError> {
        self.ensure_initialized().await?;
        let _one_at_a_time = self.history_lock.lock().await;
        Ok(history_store::clear().await?)
    }

    async fn list_installed_models(&self) -> Result<Vec<ModelInfo>, PlatformError> {
        self.ensure_initialized().await?;

        let mut installed = Vec::new();
        for spec in self.model_manager.models() {
            if self.model_manager.is_model_installed(&spec.id).await? {
                installed.push(model_info(spec, true));
            }
        }
        Ok(installed)
    }

    async fn list_available_models(&self) -> Result<Vec<ModelInfo>, PlatformError> {
        self.ensure_initialized().await?;

        let mut available = Vec::with_capacity(self.model_manager.models().len());
        for spec in self.model_manager.models() {
            let is_installed = self.model_manager.is_model_installed(&spec.id).await?;
            available.push(model_info(spec, is_installed));
        }
        Ok(available)
    }

    async fn fetch_model(
        &self,
        model_id: &str,
        token: Option<HuggingFaceToken>,
    ) -> Result<ModelProgressStream, PlatformError> {
        self.ensure_initialized().await?;

        let spec = self.model_manager.model(model_id)?.clone();
        let policy = self.load_settings().await?.performance.accel_policy();
        let (tx, rx) = mpsc::unbounded_channel();
        spawn_local(run_preload(
            Arc::clone(&self.worker),
            spec,
            token,
            policy,
            tx,
        ));
        Ok(Box::pin(UnboundedReceiverStream::new(rx)))
    }

    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo, PlatformError> {
        self.ensure_initialized().await?;

        let spec = self.model_manager.model(model_id)?;
        let is_installed = self.model_manager.is_model_installed(model_id).await?;
        Ok(model_info(spec, is_installed))
    }

    async fn remove_model(&self, model_id: &str) -> Result<(), PlatformError> {
        self.ensure_initialized().await?;
        self.model_manager.remove_model(model_id).await?;
        self.worker.release_idle();
        Ok(())
    }

    async fn verify_model(
        &self,
        model_id: &str,
        progress: ModelCheckSender,
    ) -> Result<ModelIntegrity, PlatformError> {
        self.ensure_initialized().await?;

        let spec = self.model_manager.model(model_id)?;
        match model_cache::verify_with_progress(spec, progress).await {
            Ok(()) => Ok(ModelIntegrity::Verified),
            Err(WebError::Integrity(problem)) => Ok(problem.into()),
            Err(error) => Err(error.into()),
        }
    }

    async fn is_model_installed(&self, model_id: &str) -> Result<bool, PlatformError> {
        self.ensure_initialized().await?;
        Ok(self.model_manager.is_model_installed(model_id).await?)
    }

    async fn get_model_path(&self, model_id: &str) -> Result<String, PlatformError> {
        self.ensure_initialized().await?;

        let spec = self.model_manager.model(model_id)?;
        if !self.model_manager.is_model_installed(model_id).await? {
            return Err(PlatformError::model_not_found(model_id));
        }
        Ok(model_cache::location(spec))
    }

    async fn list_available_devices(&self) -> Result<Vec<DeviceInfo>, PlatformError> {
        let mut devices = vec![DeviceInfo {
            id: 0,
            name: "CPU".to_owned(),
            description: Some("CPU-based transcription".to_owned()),
            device_type: DeviceType::Cpu,
            memory_free: None,
            memory_total: None,
            capabilities: None,
        }];
        if let Some(gpu) = run_local(detect_webgpu).await? {
            devices.push(gpu);
        }
        Ok(devices)
    }

    async fn system_report(&self) -> Result<SystemReport, PlatformError> {
        Err(PlatformError::unsupported("a system report in the browser"))
    }

    async fn load_settings(&self) -> Result<Settings, PlatformError> {
        let Some(json) = Self::get_storage_value(SETTINGS_KEY)? else {
            tracing::info!("No settings found in localStorage, using defaults");
            return Ok(Settings::default());
        };
        serde_json::from_str(&json).map_err(|e| {
            PlatformError::settings_persistence(format!(
                "The settings stored in this browser are unreadable ({e}); reset them in Settings"
            ))
        })
    }

    async fn save_settings(&self, settings: &Settings) -> Result<(), PlatformError> {
        let validation_errors = settings.validate();
        if !validation_errors.is_empty() {
            let messages: Vec<String> = validation_errors
                .into_iter()
                .map(|e| e.to_string())
                .collect();
            return Err(PlatformError::settings_persistence(format!(
                "Invalid settings: {}",
                messages.join(", ")
            )));
        }

        let json = serde_json::to_string_pretty(settings).map_err(|e| {
            PlatformError::settings_persistence(format!("Failed to serialize settings: {e}"))
        })?;
        Self::set_storage_value(SETTINGS_KEY, &json)?;
        tracing::info!("Settings saved to localStorage");
        Ok(())
    }

    async fn get_settings_location(&self) -> Result<Option<String>, PlatformError> {
        Ok(Some(format!("localStorage:{SETTINGS_KEY}")))
    }
}

async fn detect_webgpu() -> WebResult<Option<DeviceInfo>> {
    let window = web_sys::window().ok_or_else(|| {
        WebError::web_api(
            "Window",
            wasm_bindgen::JsValue::from_str("no window available"),
        )
    })?;
    let gpu = window.navigator().gpu();
    if gpu.is_undefined() {
        tracing::info!("WebGPU is not available in this browser");
        return Ok(None);
    }
    let adapter = JsFuture::from(gpu.request_adapter())
        .await
        .map_err(|e| WebError::web_api("GPU.requestAdapter", e))?;
    let Some(adapter) = adapter.into_option() else {
        tracing::info!("The browser found no WebGPU adapter");
        return Ok(None);
    };
    let info = adapter.info();
    let features = adapter.features();

    let mut capabilities = HashMap::new();
    let iterator = features.keys();
    while let Ok(entry) = iterator.next() {
        if entry.done() {
            break;
        }
        if let Some(name) = entry.value().as_string() {
            capabilities.insert(name, serde_json::Value::Bool(true));
        }
    }

    let name = adapter_name(AdapterStrings {
        vendor: &info.vendor(),
        architecture: &info.architecture(),
        device: &info.device(),
        description: &info.description(),
    });
    Ok(Some(DeviceInfo {
        id: 1,
        name: name.unwrap_or_default(),
        description: Some("WebGPU-based transcription".to_owned()),
        device_type: DeviceType::Gpu,
        memory_free: None,
        memory_total: None,
        capabilities: Some(capabilities),
    }))
}
