//! Web platform implementation of the Platform trait

use crate::audio;
use crate::compute_choice::{self, GpuFailure};
use crate::error::{IntegrityError, WebError, WebResult};
use crate::local::run_local;
use crate::model::WebModelManager;
use crate::model_cache;
use crate::model_info::model_info;
use crate::model_install::{self, InstallProgress};
use crate::protocol::{WorkerEvent, WorkerModel, WorkerRequest};
use crate::status_mapper::{PreloadMapper, PreloadStep, TranscriptionMapper};
use crate::storage::WebStorage;
use crate::storage_space;
use crate::worker_client::WorkerClient;
use crate::{history_audio, history_store};
use bytes::Bytes;
use futures::StreamExt as _;
use purr_common::accel::SystemReport;
use purr_common::adapter_name::{AdapterStrings, adapter_name};
use purr_common::audio_overview::{AudioOverview, peaks};
use purr_common::browser_compute::CpuReason;
use purr_common::devices::DeviceOrder;
use purr_common::history::{
    audio::{AudioOutcome, HistoryAudio},
    entry::{HistoryEntry, HistoryId, HistoryRecord},
    listing::HistoryListing,
};
use purr_common::hugging_face_token::HuggingFaceToken;
use purr_common::language::{tag::LanguageTag, whisper};
use purr_common::platform::{
    DeviceInfo, DeviceType, FileId, FileSource, ModelCheckSender, ModelFailureKind, ModelInfo,
    ModelIntegrity, ModelOperationProgress, ModelProgressStream, Platform, PlatformError,
    TranscriptionRequest, TranscriptionStatus, TranscriptionStream,
};
use purr_common::settings::Settings;
use purr_common::web_model::registry::{WebModel, WebModelSpec};
use purr_common::web_model::variant::ShaderF16;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, OnceCell, RwLock, mpsc};
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
    shader_f16: OnceCell<ShaderF16>,
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
        match history_audio::sweep().await {
            Ok(0) => {}
            Ok(removed) => tracing::info!("Removed {removed} stored audio files without an entry"),
            Err(error) => tracing::warn!("Could not remove orphaned stored audio: {error}"),
        }

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

    async fn shader_f16(&self) -> Result<ShaderF16, PlatformError> {
        let support = self
            .shader_f16
            .get_or_try_init(|| run_local(detect_shader_f16))
            .await?;
        Ok(*support)
    }

    /// The variant of `model` that a run would use, which is what the Models page shows.
    async fn shown_variant<'a>(
        &self,
        model: &'a WebModel,
    ) -> Result<&'a WebModelSpec, PlatformError> {
        let devices = self.load_settings().await?.performance.device_order();
        let shader_f16 = self.shader_f16().await?;
        Ok(compute_choice::shown_variant(model, &devices, shader_f16))
    }

    pub fn new() -> Result<Self, PlatformError> {
        let storage = Arc::new(WebStorage::new());
        let model_manager = Arc::new(WebModelManager::new()?);
        let worker = Arc::new(WorkerClient::spawn(crate::worker_url()));
        Ok(Self {
            storage,
            model_manager,
            worker,
            shader_f16: OnceCell::new(),
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
    if matches!(error, WebError::Integrity { .. }) {
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
    if matches!(
        error,
        WebError::Integrity {
            source: IntegrityError::Missing { .. }
        }
    ) {
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
    model: WebModel,
    shader_f16: ShaderF16,
    bytes: Bytes,
    language: Option<LanguageTag>,
    hugging_face_token: Option<HuggingFaceToken>,
    devices: DeviceOrder,
}

fn init_failed(component: &str, reason: String, details: Option<&str>) -> TranscriptionStatus {
    TranscriptionStatus::InitFailed {
        component: component.to_owned(),
        reason,
        error_details: details.map(str::to_owned),
    }
}

async fn run_transcription(
    job: Transcription,
    tx: mpsc::UnboundedSender<Result<TranscriptionStatus, PlatformError>>,
) {
    let Transcription {
        decode_lock,
        worker,
        model,
        shader_f16,
        bytes,
        language,
        hugging_face_token,
        devices,
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
        model.id
    );

    let mut choice = match compute_choice::choose(&model, &devices, shader_f16) {
        Ok(choice) => choice,
        Err(reason) => {
            send_status(&tx, cannot_run(reason));
            return;
        }
    };
    let samples = decoded.samples;
    loop {
        let Some(spec) = model.variant(choice.variant) else {
            send_status(
                &tx,
                init_failed(
                    "Model",
                    format!("{} has no {} variant", model.id, choice.variant),
                    None,
                ),
            );
            return;
        };
        let cached = match model_cache::is_installed(spec).await {
            Ok(cached) => cached,
            Err(error) => {
                send_status(&tx, init_failed("Model", error.to_string(), None));
                return;
            }
        };
        let mut mapper = TranscriptionMapper::new(spec, cached);
        if !cached {
            let mut report = |step: InstallProgress<'_>| {
                let status = match step {
                    InstallProgress::Receiving { path, loaded_bytes } => {
                        mapper.receiving(path, loaded_bytes)
                    }
                    InstallProgress::Installed { path } => mapper.installed(path),
                };
                if !send_status(&tx, status) {
                    tracing::debug!("Nobody follows the install of {} any more", spec.id);
                }
            };
            let installing = model_install::install(spec, hugging_face_token.as_ref(), &mut report);
            let installed = tokio::select! {
                () = tx.closed() => {
                    tracing::info!("Transcription stream dropped; cancelling");
                    return;
                }
                result = installing => result,
            };
            if let Err(error) = installed {
                send_status(
                    &tx,
                    init_failed(
                        "Model",
                        error.to_string(),
                        Some(
                            "Check your internet connection and try again. Files that were already downloaded are kept.",
                        ),
                    ),
                );
                return;
            }
            mapper = TranscriptionMapper::new(spec, true);
        }
        tracing::info!(
            "Transcribing with {} ({}) on {:?}",
            spec.id,
            spec.variant,
            choice.compute
        );
        if !send_status(
            &tx,
            TranscriptionStatus::BrowserCompute {
                compute: choice.compute,
            },
        ) {
            return;
        }
        let request = WorkerRequest::Transcribe {
            request_id: worker.next_request_id(),
            model: WorkerModel::from(spec),
            provider: choice.provider(),
            language: language.as_ref().map(|tag| whisper::code(tag).to_owned()),
            english_only: spec.english_only(),
        };
        let mut events = match worker.submit(request, Some(samples.clone())) {
            Ok(events) => events,
            Err(error) => {
                send_status(&tx, init_failed("Worker", error.to_string(), None));
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
    choice: &compute_choice::Choice,
) -> WorkerRequest {
    WorkerRequest::LoadModel {
        request_id: worker.next_request_id(),
        model: WorkerModel::from(spec),
        provider: choice.provider(),
    }
}

type PreloadSender = mpsc::UnboundedSender<Result<ModelOperationProgress, PlatformError>>;

fn failed(
    model_id: &str,
    operation: &str,
    error_message: String,
    kind: ModelFailureKind,
) -> ModelOperationProgress {
    ModelOperationProgress::Failed {
        model_id: model_id.to_owned(),
        operation: operation.to_owned(),
        error_message,
        kind,
    }
}

fn send_progress(tx: &PreloadSender, model_id: &str, progress: ModelOperationProgress) -> bool {
    if tx.send(Ok(progress)).is_err() {
        tracing::debug!("Nobody follows the preload of {model_id} any more");
        return false;
    }
    true
}

enum Preparation {
    Verified,
    Missing,
    Stopped,
}

/// Verifies the variant if it is installed, so a preload of a cached model also checks it.
async fn prepare(spec: &WebModelSpec, tx: &PreloadSender) -> Preparation {
    let model_id = spec.id.as_str();
    match model_cache::is_installed(spec).await {
        Ok(true) => {
            let installing = ModelOperationProgress::Installing {
                model_id: model_id.to_owned(),
            };
            if !send_progress(tx, model_id, installing) {
                return Preparation::Stopped;
            }
            match model_cache::verify(spec).await {
                Ok(()) => Preparation::Verified,
                Err(error) => {
                    let failure = failed(
                        model_id,
                        "verification",
                        error.to_string(),
                        verification_failure_kind(&error),
                    );
                    send_progress(tx, model_id, failure);
                    Preparation::Stopped
                }
            }
        }
        Ok(false) => Preparation::Missing,
        Err(error) => {
            let failure = failed(
                model_id,
                "verification",
                error.to_string(),
                ModelFailureKind::Other,
            );
            send_progress(tx, model_id, failure);
            Preparation::Stopped
        }
    }
}

/// Installs the variant, reporting progress; false when the preload is over.
async fn install_variant(
    spec: &WebModelSpec,
    token: Option<&HuggingFaceToken>,
    tx: &PreloadSender,
    mapper: &mut PreloadMapper,
) -> bool {
    let model_id = spec.id.as_str();
    let mut report = |step: InstallProgress<'_>| {
        let progress = match step {
            InstallProgress::Receiving { path, loaded_bytes } => {
                mapper.receiving(path, loaded_bytes, now_ms())
            }
            InstallProgress::Installed { path } => mapper.installed(path, now_ms()),
        };
        send_progress(tx, model_id, progress);
    };
    let installing = model_install::install(spec, token, &mut report);
    let installed = tokio::select! {
        () = tx.closed() => {
            tracing::info!("Preload stream dropped; cancelling");
            return false;
        }
        result = installing => result,
    };
    if let Err(error) = installed {
        let kind = if matches!(error, WebError::Integrity { .. }) {
            ModelFailureKind::Integrity
        } else {
            download_failure_kind()
        };
        send_progress(
            tx,
            model_id,
            failed(model_id, "download", error.to_string(), kind),
        );
        return false;
    }
    true
}

async fn run_preload(
    worker: Arc<WorkerClient>,
    model: WebModel,
    shader_f16: ShaderF16,
    hugging_face_token: Option<HuggingFaceToken>,
    devices: DeviceOrder,
    tx: PreloadSender,
) {
    let model_id = model.id.clone();
    let fail = |operation: &str, error_message: String, kind: ModelFailureKind| {
        failed(&model_id, operation, error_message, kind)
    };
    let starting = ModelOperationProgress::Starting {
        model_id: model_id.clone(),
    };
    if !send_progress(&tx, &model_id, starting) {
        return;
    }

    let mut choice = match compute_choice::choose(&model, &devices, shader_f16) {
        Ok(choice) => choice,
        Err(reason) => {
            let failure = fail(
                "download",
                reason.gpu_only_refusal().to_owned(),
                ModelFailureKind::Other,
            );
            send_progress(&tx, &model_id, failure);
            return;
        }
    };
    let Some(mut spec) = model.variant(choice.variant) else {
        let failure = fail(
            "download",
            format!("{model_id} has no {} variant", choice.variant),
            ModelFailureKind::Other,
        );
        send_progress(&tx, &model_id, failure);
        return;
    };
    let mut verified_first = match prepare(spec, &tx).await {
        Preparation::Verified => true,
        Preparation::Missing => false,
        Preparation::Stopped => return,
    };

    let mut mapper = PreloadMapper::new(spec, now_ms());
    if !verified_first
        && !install_variant(spec, hugging_face_token.as_ref(), &tx, &mut mapper).await
    {
        return;
    }
    let mut events = match worker.submit(load_request(&worker, spec, &choice), None) {
        Ok(events) => events,
        Err(error) => {
            let failure = fail("download", error.to_string(), ModelFailureKind::Other);
            send_progress(&tx, &model_id, failure);
            return;
        }
    };

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
                            let Some(cpu_spec) = model.variant(choice.variant) else {
                                let failure = fail(
                                    "download",
                                    format!("{model_id} has no {} variant", choice.variant),
                                    ModelFailureKind::Other,
                                );
                                send_progress(&tx, &model_id, failure);
                                return;
                            };
                            if cpu_spec.variant != spec.variant {
                                spec = cpu_spec;
                                mapper = PreloadMapper::new(spec, now_ms());
                                verified_first = match prepare(spec, &tx).await {
                                    Preparation::Verified => true,
                                    Preparation::Missing => false,
                                    Preparation::Stopped => return,
                                };
                                if !verified_first
                                    && !install_variant(spec, hugging_face_token.as_ref(), &tx, &mut mapper).await
                                {
                                    return;
                                }
                            }
                            match worker.submit(load_request(&worker, spec, &choice), None) {
                                Ok(retry) => {
                                    events = retry;
                                    continue;
                                }
                                Err(error) => {
                                    let failure = fail("download", error.to_string(), ModelFailureKind::Other);
                                    send_progress(&tx, &model_id, failure);
                                    return;
                                }
                            }
                        }
                        GpuFailure::CannotRun(reason) => {
                            worker.release_idle();
                            let failure = fail("download", reason.gpu_only_refusal().to_owned(), ModelFailureKind::Other);
                            send_progress(&tx, &model_id, failure);
                            return;
                        }
                        GpuFailure::Unrelated => {}
                    }
                }
                let outcome = match mapper.map(event, now_ms()) {
                    PreloadStep::Progress(progress) => progress,
                    PreloadStep::Ignored => continue,
                    PreloadStep::Failed(message) => {
                        worker.release_idle();
                        fail("download", message, download_failure_kind())
                    }
                    PreloadStep::Loaded => {
                        if !verified_first {
                            let installing = ModelOperationProgress::Installing { model_id: model_id.clone() };
                            if !send_progress(&tx, &model_id, installing) {
                                return;
                            }
                        }
                        let verified = if verified_first {
                            Ok(())
                        } else {
                            model_cache::verify(spec).await
                        };
                        match verified {
                            Ok(()) => ModelOperationProgress::Completed {
                                model_id: model_id.clone(),
                                local_path: PathBuf::from(model_cache::location(spec)),
                            },
                            Err(error) => {
                                worker.release_idle();
                                let (message, kind) = downloaded_check_failure(spec, &error).await;
                                fail("verification", message, kind)
                            }
                        }
                    }
                };
                let finished = matches!(
                    outcome,
                    ModelOperationProgress::Completed { .. } | ModelOperationProgress::Failed { .. }
                );
                if !send_progress(&tx, &model_id, outcome) || finished {
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

        let model = self.model_manager.model(request.model.as_str())?.clone();
        let shader_f16 = self.shader_f16().await?;
        let language = request.language;

        tracing::info!(
            "Transcribing {} bytes with {}, language {:?}",
            bytes.len(),
            model.id,
            language
        );

        let (tx, rx) = mpsc::unbounded_channel();
        spawn_local(run_transcription(
            Transcription {
                decode_lock: Arc::clone(&self.decode_lock),
                worker: Arc::clone(&self.worker),
                model,
                shader_f16,
                bytes,
                language,
                hugging_face_token: request.hugging_face_token,
                devices: request.compute.devices,
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

    async fn save_history_entry(
        &self,
        entry: &HistoryEntry,
        audio: Option<HistoryAudio>,
    ) -> Result<AudioOutcome, PlatformError> {
        self.ensure_initialized().await?;
        let _one_at_a_time = self.history_lock.lock().await;
        Ok(history_store::save(entry, audio).await?)
    }

    async fn history_audio(&self, id: &HistoryId) -> Result<Option<HistoryAudio>, PlatformError> {
        self.ensure_initialized().await?;
        let _one_at_a_time = self.history_lock.lock().await;
        Ok(history_audio::read(id).await?)
    }

    async fn history_audio_bytes(&self) -> Result<u64, PlatformError> {
        self.ensure_initialized().await?;
        let _one_at_a_time = self.history_lock.lock().await;
        Ok(history_audio::used_bytes().await?)
    }

    async fn clear_history_audio(&self) -> Result<u64, PlatformError> {
        self.ensure_initialized().await?;
        let _one_at_a_time = self.history_lock.lock().await;
        Ok(history_audio::clear().await?)
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
        for model in self.model_manager.models() {
            let spec = self.shown_variant(model).await?;
            if self.model_manager.is_installed(spec).await? {
                installed.push(model_info(model, spec, true));
            }
        }
        Ok(installed)
    }

    async fn list_available_models(&self) -> Result<Vec<ModelInfo>, PlatformError> {
        self.ensure_initialized().await?;

        let mut available = Vec::with_capacity(self.model_manager.models().len());
        for model in self.model_manager.models() {
            let spec = self.shown_variant(model).await?;
            let is_installed = self.model_manager.is_installed(spec).await?;
            available.push(model_info(model, spec, is_installed));
        }
        Ok(available)
    }

    async fn fetch_model(
        &self,
        model_id: &str,
        token: Option<HuggingFaceToken>,
    ) -> Result<ModelProgressStream, PlatformError> {
        self.ensure_initialized().await?;

        let model = self.model_manager.model(model_id)?.clone();
        let devices = self.load_settings().await?.performance.device_order();
        let shader_f16 = self.shader_f16().await?;
        let (tx, rx) = mpsc::unbounded_channel();
        spawn_local(run_preload(
            Arc::clone(&self.worker),
            model,
            shader_f16,
            token,
            devices,
            tx,
        ));
        Ok(Box::pin(UnboundedReceiverStream::new(rx)))
    }

    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo, PlatformError> {
        self.ensure_initialized().await?;

        let model = self.model_manager.model(model_id)?;
        let spec = self.shown_variant(model).await?;
        let is_installed = self.model_manager.is_installed(spec).await?;
        Ok(model_info(model, spec, is_installed))
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

        let model = self.model_manager.model(model_id)?;
        let spec = self.shown_variant(model).await?;
        match model_cache::verify_with_progress(spec, progress).await {
            Ok(()) => Ok(ModelIntegrity::Verified),
            Err(WebError::Integrity { source: problem }) => Ok(problem.into()),
            Err(error) => Err(error.into()),
        }
    }

    async fn is_model_installed(&self, model_id: &str) -> Result<bool, PlatformError> {
        self.ensure_initialized().await?;
        let model = self.model_manager.model(model_id)?;
        let spec = self.shown_variant(model).await?;
        Ok(self.model_manager.is_installed(spec).await?)
    }

    async fn get_model_path(&self, model_id: &str) -> Result<String, PlatformError> {
        self.ensure_initialized().await?;

        let model = self.model_manager.model(model_id)?;
        let spec = self.shown_variant(model).await?;
        if !self.model_manager.is_installed(spec).await? {
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
            key: None,
            seen_by: Vec::new(),
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
        key: None,
        seen_by: Vec::new(),
    }))
}

async fn detect_shader_f16() -> WebResult<ShaderF16> {
    let capable = detect_webgpu().await?.is_some_and(|gpu| {
        gpu.capabilities
            .is_some_and(|capabilities| capabilities.contains_key("shader-f16"))
    });
    Ok(if capable {
        ShaderF16::Supported
    } else {
        ShaderF16::Missing
    })
}
