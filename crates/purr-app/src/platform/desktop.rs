/// Default platform implementation for desktop/native builds
/// This implementation uses direct file system access and native whisper transcription
use super::{
    Platform, PlatformError, TranscriptionRequest, TranscriptionStatus,
    engine_options::{apply_engine, device_info, last_run, probe},
    history_store::HistoryDir,
};
use bytes::Bytes;
use futures::{SinkExt, Stream, StreamExt, channel::mpsc};
use parking_lot::Mutex;
use purr_common::accel::{LastRun, SystemReport};
use purr_common::audio_overview::{AudioOverview, PeakFold};
use purr_common::history::{
    audio::{AudioOutcome, HistoryAudio},
    entry::{HistoryEntry, HistoryId, HistoryRecord},
    listing::HistoryListing,
};
use purr_common::hugging_face_token::HuggingFaceToken;
use purr_common::platform::{
    ChunkKind, DeviceInfo, FileId, FileSource, IntegrityProblem, ModelCheckProgress,
    ModelCheckSender, ModelFailureKind, ModelInfo, ModelIntegrity, ModelMetadata,
    ModelOperationProgress, ModelProgressStream, TranscriptionStream, model_flag,
};
use purr_common::settings::Settings;
use purr_core::WhisperError;
use purr_core::config::TentativeText;
use purr_core::dev;
use purr_core::error::ConfigurationError;
use purr_core::model::attributes::{ModelFeature, ModelLanguage};
use purr_core::model::registry::ModelSpec;
use purr_core::model::status::ModelIntegrity as CoreIntegrity;
use purr_core::model::{ModelManager, vad_or_none};
use std::{
    collections::VecDeque,
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};
use tokio::sync::oneshot;
use tracing::{debug, error, info, warn};

pub type PlatformImpl = DesktopPlatformImpl;

const SETTINGS_FILE: &str = "settings.toml";

pub(super) struct DesktopPlatformImpl {
    temp_dir: PathBuf,
    model_manager: Arc<ModelManager>,
    vad_gate: Arc<tokio::sync::Mutex<()>>,
    history_lock: tokio::sync::Mutex<()>,
    last_run: Arc<Mutex<Option<LastRun>>>,
}

fn upload_dir() -> PathBuf {
    std::env::temp_dir().join("purr-temp")
}

/// How long an upload a closed session left behind is kept.
const UPLOAD_RETENTION: std::time::Duration = std::time::Duration::from_hours(24);

/// Deletes the uploads in `dir` older than [`UPLOAD_RETENTION`]; a file that resists is logged.
fn purge_old_uploads(dir: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let age = entry.metadata()?.modified()?.elapsed();
        if age.is_ok_and(|age| age > UPLOAD_RETENTION)
            && let Err(error) = std::fs::remove_file(entry.path())
        {
            warn!(
                "Could not remove the old upload {}: {error}",
                entry.path().display()
            );
        }
    }
    Ok(())
}

impl DesktopPlatformImpl {
    pub fn new() -> Result<Self, PlatformError> {
        let temp_dir = upload_dir();
        std::fs::create_dir_all(&temp_dir).map_err(PlatformError::io)?;
        if let Err(error) = purge_old_uploads(&temp_dir) {
            warn!(
                "Could not clear old uploads from {}: {error}",
                temp_dir.display()
            );
        }

        let model_manager = ModelManager::new().map_err(PlatformError::model_management)?;

        Ok(Self {
            temp_dir,
            model_manager: Arc::new(model_manager),
            vad_gate: Arc::new(tokio::sync::Mutex::new(())),
            history_lock: tokio::sync::Mutex::new(()),
            last_run: Arc::new(Mutex::new(None)),
        })
    }
}

impl Platform for DesktopPlatformImpl {
    fn install_logging_hooks(&self) {
        purr_core::whisper::logging::install_logging_hooks();
    }

    async fn process_file(
        &self,
        file_data: Bytes,
        file_path: &Path,
    ) -> Result<FileId, PlatformError> {
        use std::fs::File;
        use std::io::Write;
        use uuid::Uuid;

        let file_id: FileId = Uuid::new_v4().to_string().into();
        let stored = self.temp_dir.join(&file_id);
        info!("Storing {} as {}", file_path.display(), stored.display());

        let mut file = File::create(&stored).map_err(PlatformError::io)?;
        file.write_all(&file_data).map_err(PlatformError::io)?;
        file.flush().map_err(PlatformError::io)?;
        Ok(file_id)
    }

    async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionStream, PlatformError> {
        let audio_path = match &request.file {
            FileSource::Bytes(bytes) => {
                return Err(PlatformError::unsupported(format!(
                    "transcribing {} bytes held in memory; save them to a file first",
                    bytes.len()
                )));
            }
            FileSource::Path(path) => path.clone(),
            FileSource::Uploaded(file_id) => self.temp_dir.join(file_id),
        };
        let manager = &self.model_manager;
        let model = lookup(manager, request.model.as_str())?;
        let installed = manager
            .is_model_downloaded(model)
            .await
            .map_err(PlatformError::model_management)?;
        if !installed {
            return Err(PlatformError::model_not_found(request.model.to_string()));
        }
        let vad = vad_or_none(manager.vad_model_path().await);
        if vad.is_none() {
            self.fetch_vad_in_background(request.hugging_face_token.clone());
        }
        let config = apply_engine(
            purr_core::TranscriptionConfig::new()
                .with_translate(request.translate)
                .with_model_path(manager.get_model_path(model))
                .with_vad_model(vad)
                .with_opt_language(request.language)
                .apply_output_format(|format| format.with_word_timestamps(true))
                .with_beam_telemetry(request.beam_telemetry.clone())
                .with_tentative(TentativeText::Decoded),
            &request.engine,
        );
        info!(
            "Transcribing {} with {}",
            audio_path.display(),
            request.model
        );

        Ok(transcription_stream(
            audio_path,
            config,
            model.file.size_bytes,
            Arc::clone(&self.last_run),
        ))
    }

    async fn cleanup(&self, file_id: &str) -> Result<(), PlatformError> {
        let file_path = self.temp_dir.join(file_id);
        if file_path.exists() {
            std::fs::remove_file(&file_path).map_err(PlatformError::io)?;
            info!("Cleaned up temporary file: {}", file_id);
        }
        Ok(())
    }

    fn max_upload_bytes(&self) -> Option<u64> {
        None
    }

    async fn audio_overview(
        &self,
        file_id: &FileId,
        buckets: usize,
    ) -> Result<AudioOverview, PlatformError> {
        let mut stream = purr_core::AudioProcessor::stream_file(self.temp_dir.join(file_id))
            .await
            .map_err(PlatformError::audio_processing)?;
        let mut fold: Option<PeakFold> = None;
        let mut sample_rate = 0;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(PlatformError::audio_processing)?;
            sample_rate = chunk.sample_rate;
            fold.get_or_insert_with(|| PeakFold::new(chunk.sample_rate))
                .push(&chunk.samples);
            if chunk.is_final {
                break;
            }
        }
        let fold = fold
            .filter(|_| sample_rate > 0)
            .ok_or_else(|| PlatformError::audio_processing("the file holds no audio"))?;
        Ok(AudioOverview {
            duration_seconds: fold.samples() as f64 / f64::from(sample_rate),
            peaks: fold.finish(buckets),
        })
    }

    // ========================================
    // History Implementation
    // ========================================

    async fn list_history(&self) -> Result<HistoryListing, PlatformError> {
        self.history_task(|history| history.list()).await
    }

    async fn get_history_entry(&self, id: &HistoryId) -> Result<HistoryRecord, PlatformError> {
        let id = id.clone();
        self.history_task(move |history| history.get(&id)).await
    }

    async fn save_history_entry(
        &self,
        entry: &HistoryEntry,
        _audio: Option<HistoryAudio>,
    ) -> Result<AudioOutcome, PlatformError> {
        let entry = entry.clone();
        self.history_task(move |history| history.save(&entry))
            .await?;
        Ok(AudioOutcome::NotOffered)
    }

    async fn history_audio(&self, _id: &HistoryId) -> Result<Option<HistoryAudio>, PlatformError> {
        Ok(None)
    }

    async fn history_audio_bytes(&self) -> Result<u64, PlatformError> {
        Ok(0)
    }

    async fn clear_history_audio(&self) -> Result<u64, PlatformError> {
        Ok(0)
    }

    async fn rename_history_entry(&self, id: &HistoryId, title: &str) -> Result<(), PlatformError> {
        let (id, title) = (id.clone(), title.to_owned());
        self.history_task(move |history| history.rename(&id, &title))
            .await
    }

    async fn delete_history_entry(&self, id: &HistoryId) -> Result<(), PlatformError> {
        let id = id.clone();
        self.history_task(move |history| history.delete(&id)).await
    }

    async fn clear_history(&self) -> Result<usize, PlatformError> {
        self.history_task(|history| history.clear()).await
    }

    // ========================================
    // Model Management Implementation
    // ========================================

    async fn list_installed_models(&self) -> Result<Vec<ModelInfo>, PlatformError> {
        let manager = &self.model_manager;
        let downloaded_models = manager
            .list_downloaded_models()
            .await
            .map_err(PlatformError::model_management)?;

        Ok(downloaded_models
            .into_iter()
            .map(|model| model_info(manager, model, true))
            .collect())
    }

    async fn list_available_models(&self) -> Result<Vec<ModelInfo>, PlatformError> {
        let mut model_infos = Vec::new();
        let manager = &self.model_manager;

        for model in manager.registry().models() {
            let is_installed = manager
                .is_model_downloaded(model)
                .await
                .map_err(PlatformError::model_management)?;
            model_infos.push(model_info(manager, model, is_installed));
        }

        Ok(model_infos)
    }

    fn fetch_model(
        &self,
        model_id: &str,
        token: Option<HuggingFaceToken>,
    ) -> impl Future<Output = Result<ModelProgressStream, PlatformError>> + Send {
        std::future::ready(self.start_download(model_id, token))
    }

    async fn get_model_info(&self, model_id: &str) -> Result<ModelInfo, PlatformError> {
        let manager = &self.model_manager;
        let model = lookup(manager, model_id)?;
        let is_installed = manager
            .is_model_downloaded(model)
            .await
            .map_err(PlatformError::model_management)?;

        Ok(model_info(manager, model, is_installed))
    }

    async fn remove_model(&self, model_id: &str) -> Result<(), PlatformError> {
        let manager = &self.model_manager;
        let model = lookup(manager, model_id)?;
        manager
            .delete_model(model)
            .await
            .map_err(PlatformError::model_management)?;

        Ok(())
    }

    async fn verify_model(
        &self,
        model_id: &str,
        progress: ModelCheckSender,
    ) -> Result<ModelIntegrity, PlatformError> {
        let manager = &self.model_manager;
        let model = lookup(manager, model_id)?;
        let integrity = manager
            .verify_model_with_progress(model, move |checked_bytes, total_bytes| {
                if progress
                    .unbounded_send(ModelCheckProgress {
                        checked_bytes,
                        total_bytes,
                    })
                    .is_err()
                {
                    debug!("Nobody follows the model check any more");
                }
            })
            .await
            .map_err(PlatformError::model_management)?;
        Ok(match integrity {
            CoreIntegrity::Verified => ModelIntegrity::Verified,
            CoreIntegrity::Missing => damaged(IntegrityProblem::Missing, model),
            CoreIntegrity::SizeMismatch { .. } => damaged(IntegrityProblem::SizeMismatch, model),
            CoreIntegrity::ChecksumMismatch { .. } => {
                damaged(IntegrityProblem::ChecksumMismatch, model)
            }
        })
    }

    async fn is_model_installed(&self, model_id: &str) -> Result<bool, PlatformError> {
        let manager = &self.model_manager;
        let model = lookup(manager, model_id)?;
        manager
            .is_model_downloaded(model)
            .await
            .map_err(PlatformError::model_management)
    }

    async fn get_model_path(&self, model_id: &str) -> Result<String, PlatformError> {
        let manager = &self.model_manager;
        let model = lookup(manager, model_id)?;
        let is_installed = manager
            .is_model_downloaded(model)
            .await
            .map_err(PlatformError::model_management)?;
        if !is_installed {
            return Err(PlatformError::model_not_found(model_id.to_owned()));
        }

        let model_path = manager.get_model_path(model);
        Ok(model_path.to_string_lossy().to_string())
    }

    async fn list_available_devices(&self) -> Result<Vec<DeviceInfo>, PlatformError> {
        Ok(dev::list_devices().into_iter().map(device_info).collect())
    }

    async fn system_report(&self) -> Result<SystemReport, PlatformError> {
        let last = self.last_run.lock().clone();
        tokio::task::spawn_blocking(move || probe(last))
            .await
            .map_err(PlatformError::device_detection)
    }

    // ========================================
    // Settings Persistence Implementation
    // ========================================

    async fn load_settings(&self) -> Result<Settings, PlatformError> {
        let settings_path = Self::get_settings_file_path()?;

        debug!("Loading settings from: {}", settings_path.display());

        // If the file doesn't exist, return default settings
        if !settings_path.exists() {
            info!(
                "Settings file not found, using defaults: {}",
                settings_path.display()
            );
            return Ok(Settings::default());
        }

        // Read the settings file
        let contents = tokio::fs::read_to_string(&settings_path)
            .await
            .map_err(|e| {
                error!(
                    "Failed to read settings file {}: {}",
                    settings_path.display(),
                    e
                );
                PlatformError::settings_persistence(e)
            })?;

        // Parse TOML content
        let settings: Settings = toml::from_str(&contents).map_err(|e| {
            let reason = describe_parse_error(&contents, &e);
            error!(
                "Failed to parse settings file {}: {reason}",
                settings_path.display()
            );
            PlatformError::settings_persistence(reason)
        })?;

        info!(
            "Successfully loaded settings from: {}",
            settings_path.display()
        );
        Ok(settings)
    }

    async fn save_settings(&self, settings: &Settings) -> Result<(), PlatformError> {
        let settings_dir = Self::get_settings_dir()?;
        let settings_path = settings_dir.join(SETTINGS_FILE);

        debug!("Saving settings to: {}", settings_path.display());

        tokio::fs::create_dir_all(&settings_dir)
            .await
            .map_err(|e| {
                error!(
                    "Failed to create settings directory {}: {}",
                    settings_dir.display(),
                    e
                );
                PlatformError::settings_persistence(e)
            })?;

        let toml_content = toml::to_string_pretty(settings).map_err(|e| {
            error!("Failed to serialize settings to TOML: {}", e);
            PlatformError::settings_persistence(e)
        })?;

        let temp_path = settings_dir.join(format!("{SETTINGS_FILE}.{}.tmp", uuid::Uuid::new_v4()));

        tokio::fs::write(&temp_path, &toml_content)
            .await
            .map_err(|e| {
                error!(
                    "Failed to write temporary settings file {}: {}",
                    temp_path.display(),
                    e
                );
                PlatformError::settings_persistence(e)
            })?;

        if let Err(e) = tokio::fs::rename(&temp_path, &settings_path).await {
            error!(
                "Failed to rename settings file from {} to {}: {}",
                temp_path.display(),
                settings_path.display(),
                e
            );
            if let Err(cleanup) = tokio::fs::remove_file(&temp_path).await {
                error!(
                    "Failed to remove the temporary settings file {}: {}",
                    temp_path.display(),
                    cleanup
                );
            }
            return Err(PlatformError::settings_persistence(e));
        }

        info!(
            "Successfully saved settings to: {}",
            settings_path.display()
        );
        Ok(())
    }

    async fn get_settings_location(&self) -> Result<Option<String>, PlatformError> {
        let settings_path = Self::get_settings_file_path()?;
        Ok(Some(settings_path.to_string_lossy().to_string()))
    }
}

fn lookup<'a>(manager: &'a ModelManager, model_id: &str) -> Result<&'a ModelSpec, PlatformError> {
    manager
        .registry()
        .get(model_id)
        .map_err(|err| PlatformError::model_not_found(err.model_name))
}

/// Where and why the file failed to parse, without the line of the file that holds the token.
fn describe_parse_error(contents: &str, error: &toml::de::Error) -> String {
    let line = error
        .span()
        .and_then(|span| contents.get(..span.start))
        .map(|before| before.matches('\n').count() + 1);
    line.map_or_else(
        || error.message().to_owned(),
        |line| format!("{} (line {line})", error.message()),
    )
}

fn lookup_shared(manager: &ModelManager, model_id: &str) -> Result<Arc<ModelSpec>, PlatformError> {
    manager
        .registry()
        .get_shared(model_id)
        .map_err(|err| PlatformError::model_not_found(err.model_name))
}

fn damaged(problem: IntegrityProblem, model: &ModelSpec) -> ModelIntegrity {
    ModelIntegrity::Damaged {
        problem,
        file: model.filename.clone(),
    }
}

/// Ends the download task when the progress stream is dropped.
struct CancelOnDrop<S> {
    progress: S,
    _cancel: oneshot::Sender<()>,
}

impl<S: Stream + Unpin> Stream for CancelOnDrop<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.progress).poll_next(cx)
    }
}

type ProgressSender = mpsc::Sender<Result<ModelOperationProgress, PlatformError>>;

type Status = Result<TranscriptionStatus, PlatformError>;

/// Where a desktop transcription stream is.
enum Phase {
    Load {
        audio_path: PathBuf,
        config: purr_core::TranscriptionConfig,
    },
    Decode(purr_core::StreamingTranscription),
    Finished,
}

/// One desktop transcription, folded into the statuses the UI reads.
struct DesktopRun {
    phase: Phase,
    pending: VecDeque<Status>,
    last_run: Arc<Mutex<Option<LastRun>>>,
    started: Instant,
    language_reported: bool,
    word_count: usize,
    audio_duration: f32,
    last_end: f64,
}

impl DesktopRun {
    async fn advance(&mut self) {
        match std::mem::replace(&mut self.phase, Phase::Finished) {
            Phase::Load { audio_path, config } => {
                match purr_core::transcribe_file_stream(&audio_path, config).await {
                    Ok(stream) => {
                        let ran = last_run(stream.accel());
                        *self.last_run.lock() = Some(ran.clone());
                        self.pending
                            .push_back(Ok(TranscriptionStatus::Engine { run: ran }));
                        self.pending
                            .push_back(Ok(TranscriptionStatus::Transcribing));
                        self.phase = Phase::Decode(stream);
                    }
                    Err(error) => {
                        error!("Failed to start transcription: {error}");
                        self.pending
                            .push_back(Err(PlatformError::transcription(error)));
                    }
                }
            }
            Phase::Decode(mut stream) => match stream.next().await {
                Some(Ok(purr_core::StreamEvent::Progress(progress))) => {
                    self.pending
                        .push_back(Ok(TranscriptionStatus::Progress(progress.frame)));
                    self.phase = Phase::Decode(stream);
                }
                Some(Ok(purr_core::StreamEvent::Beams(snapshot))) => {
                    self.pending
                        .push_back(Ok(TranscriptionStatus::Telemetry(snapshot)));
                    self.phase = Phase::Decode(stream);
                }
                Some(Ok(purr_core::StreamEvent::Chunk(chunk))) => {
                    if !self.language_reported
                        && let Some(language) = chunk.language.clone()
                    {
                        self.language_reported = true;
                        self.pending
                            .push_back(Ok(TranscriptionStatus::LanguageDetected { language }));
                    }
                    match chunk.kind {
                        ChunkKind::Committed => {
                            self.word_count += chunk.text.split_whitespace().count();
                            self.last_end = self.last_end.max(chunk.end);
                        }
                        ChunkKind::Tentative => {}
                    }
                    if let Some(stats) = &chunk.final_stats {
                        self.audio_duration = stats.audio_duration;
                    }
                    self.pending.push_back(Ok(TranscriptionStatus::InProgress {
                        chunk_index: chunk.chunk_index,
                        text: chunk.text,
                        start_time: chunk.start,
                        end_time: chunk.end,
                        words: chunk.words,
                        kind: chunk.kind,
                    }));
                    self.phase = Phase::Decode(stream);
                }
                Some(Err(error)) => {
                    error!("Transcription error: {error}");
                    self.pending
                        .push_back(Err(PlatformError::transcription(error)));
                }
                None => {
                    let processing_time = self.started.elapsed().as_secs_f64();
                    info!("Transcription completed in {processing_time:.2}s");
                    self.pending.push_back(Ok(TranscriptionStatus::Completed {
                        processing_time,
                        audio_duration: if self.audio_duration > 0.0 {
                            self.audio_duration
                        } else {
                            self.last_end as f32
                        },
                        word_count: self.word_count,
                    }));
                }
            },
            Phase::Finished => {}
        }
    }
}

/// The statuses of one transcription, produced as the caller polls them.
///
/// Whisper runs blocking inside a runtime worker, so the core stream is polled by the caller
/// rather than by a forwarding task that could starve on that worker. Dropping the stream drops
/// the core stream, which stops Whisper after the chunk it is working on; a model load that
/// already started finishes first.
fn transcription_stream(
    audio_path: PathBuf,
    config: purr_core::TranscriptionConfig,
    model_bytes: u64,
    last_run: Arc<Mutex<Option<LastRun>>>,
) -> TranscriptionStream {
    let run = DesktopRun {
        phase: Phase::Load { audio_path, config },
        pending: VecDeque::from([
            Ok(TranscriptionStatus::Starting),
            Ok(TranscriptionStatus::LoadingModel {
                loaded_bytes: 0,
                total_bytes: model_bytes,
            }),
        ]),
        last_run,
        started: Instant::now(),
        language_reported: false,
        word_count: 0,
        audio_duration: 0.0,
        last_end: 0.0,
    };
    Box::pin(futures::stream::unfold(run, |mut run| async move {
        loop {
            if let Some(status) = run.pending.pop_front() {
                return Some((status, run));
            }
            if matches!(run.phase, Phase::Finished) {
                return None;
            }
            run.advance().await;
        }
    }))
}

fn failure_kind(error: &WhisperError) -> ModelFailureKind {
    let network = matches!(
        error,
        WhisperError::Configuration {
            source: ConfigurationError::DownloadFailed { .. }
                | ConfigurationError::DownloadStreamError { .. },
        } | WhisperError::Http { .. }
            | WhisperError::NetworkError { .. }
    );
    let integrity = matches!(
        error,
        WhisperError::Configuration {
            source: ConfigurationError::ModelIntegrity(_),
        }
    );
    if network {
        ModelFailureKind::Network
    } else if integrity {
        ModelFailureKind::Integrity
    } else {
        ModelFailureKind::Other
    }
}

async fn emit(tx: &mut ProgressSender, progress: ModelOperationProgress) -> bool {
    if tx.send(Ok(progress)).await.is_ok() {
        true
    } else {
        debug!("The model progress receiver is gone");
        false
    }
}

async fn run_download(
    manager: Arc<ModelManager>,
    model: Arc<ModelSpec>,
    token: Option<HuggingFaceToken>,
    mut tx: ProgressSender,
    cancelled: oneshot::Receiver<()>,
) {
    let outcome = tokio::select! {
        biased;
        _ = cancelled => {
            info!("Download of {} cancelled", model.id);
            discard_partial(&manager, &model).await;
            return;
        }
        outcome = download(&manager, &model, token.as_ref(), &mut tx) => outcome,
    };
    if let Err(failure) = outcome {
        emit(&mut tx, failure).await;
    }
}

async fn discard_partial(manager: &ModelManager, model: &ModelSpec) {
    let partial = manager.get_model_path(model).with_extension("tmp");
    if let Err(error) = tokio::fs::remove_file(&partial).await
        && error.kind() != std::io::ErrorKind::NotFound
    {
        warn!(
            "Could not remove the partial download {}: {error}",
            partial.display()
        );
    }
}

/// Downloads `model`, reporting progress on `tx`; the `Err` is the terminal failure to report.
async fn download(
    manager: &ModelManager,
    model: &ModelSpec,
    token: Option<&HuggingFaceToken>,
    tx: &mut ProgressSender,
) -> Result<(), ModelOperationProgress> {
    let model_id = model.id.clone();
    let failed = |operation: &str, error: &WhisperError| ModelOperationProgress::Failed {
        model_id: model_id.clone(),
        operation: operation.to_owned(),
        error_message: error.to_string(),
        kind: failure_kind(error),
    };

    let starting = ModelOperationProgress::Starting {
        model_id: model_id.clone(),
    };
    if !emit(tx, starting).await {
        return Ok(());
    }

    let installed = manager
        .is_model_downloaded(model)
        .await
        .map_err(|error| failed("model_check", &error))?;
    if installed {
        let completed = ModelOperationProgress::Completed {
            model_id: model_id.clone(),
            local_path: manager.get_model_path(model),
        };
        emit(tx, completed).await;
        return Ok(());
    }

    let started = Instant::now();
    let mut progress_tx = tx.clone();
    let local_path = manager
        .download_model_with_progress(model, token, |downloaded, total| {
            let seconds = started.elapsed().as_secs_f64();
            let progress = ModelOperationProgress::Downloading {
                model_id: model_id.clone(),
                bytes_downloaded: downloaded,
                total_bytes: total,
                speed_bps: (seconds > 0.0).then(|| (downloaded as f64 / seconds) as u64),
            };
            if progress_tx.try_send(Ok(progress)).is_err() {
                debug!("Skipped a progress update of {model_id}");
            }
        })
        .await
        .map_err(|error| failed("model_loading", &error))?;

    let completed = ModelOperationProgress::Completed {
        model_id: model_id.clone(),
        local_path,
    };
    emit(tx, completed).await;
    Ok(())
}

async fn fetch_vad(
    manager: Arc<ModelManager>,
    gate: Arc<tokio::sync::Mutex<()>>,
    token: Option<HuggingFaceToken>,
) {
    let _one_at_a_time = gate.lock().await;
    let fetched = manager.ensure_vad_model(token.as_ref()).await.map(Some);
    if let Some(path) = vad_or_none(fetched) {
        debug!("The voice activity model is at {}", path.display());
    }
}

fn model_info(manager: &ModelManager, model: &ModelSpec, is_installed: bool) -> ModelInfo {
    let mut info = ModelInfo::new(
        model.id.clone(),
        model.id.clone(),
        model.description.clone(),
        model.file.size_bytes,
        is_installed,
        model_metadata(model),
    );
    if is_installed {
        info.mark_installed(manager.get_model_path(model));
    }
    info
}

fn model_metadata(model: &ModelSpec) -> ModelMetadata {
    let language_support = match model.language {
        ModelLanguage::Multilingual => "multilingual",
        ModelLanguage::EnglishOnly => "en-only",
    };
    let mut metadata = ModelMetadata::new(
        "ggml".to_owned(),
        "whisper".to_owned(),
        language_support.to_owned(),
        "v1".to_owned(),
    )
    .with_download_url(model.url().to_owned())
    .with_quantization(model.quantization.to_string());
    metadata
        .platform_specific
        .insert("sha256".to_owned(), model.file.sha256.to_string());
    metadata
        .platform_specific
        .insert("family".to_owned(), model.family.to_string());
    metadata.platform_specific.insert(
        model_flag::ENGLISH_ONLY.to_owned(),
        matches!(model.language, ModelLanguage::EnglishOnly).to_string(),
    );
    metadata.platform_specific.insert(
        model_flag::SPEAKER_TURNS.to_owned(),
        model.has_feature(ModelFeature::Tdrz).to_string(),
    );
    metadata
}

impl DesktopPlatformImpl {
    fn history_dir() -> Result<HistoryDir, PlatformError> {
        let dirs = directories::ProjectDirs::from("dev.gaetans", "", "purr").ok_or_else(|| {
            PlatformError::history(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not determine the data directory",
            ))
        })?;
        Ok(HistoryDir::new(dirs.data_dir().join("history")))
    }

    async fn history_task<T, E>(
        &self,
        work: impl FnOnce(HistoryDir) -> Result<T, E> + Send + 'static,
    ) -> Result<T, PlatformError>
    where
        T: Send + 'static,
        E: Into<PlatformError> + Send + 'static,
    {
        let history = Self::history_dir()?;
        let _one_at_a_time = self.history_lock.lock().await;
        tokio::task::spawn_blocking(move || work(history))
            .await
            .map_err(PlatformError::history)?
            .map_err(Into::into)
    }

    /// Fetches the voice activity model without holding anything up; a failure is only logged.
    fn fetch_vad_in_background(&self, token: Option<HuggingFaceToken>) {
        tokio::spawn(fetch_vad(
            Arc::clone(&self.model_manager),
            Arc::clone(&self.vad_gate),
            token,
        ));
    }

    fn start_download(
        &self,
        model_id: &str,
        token: Option<HuggingFaceToken>,
    ) -> Result<ModelProgressStream, PlatformError> {
        let model = lookup_shared(&self.model_manager, model_id)?;
        let manager = Arc::clone(&self.model_manager);
        let (tx, rx) = mpsc::channel(10);
        let (cancel, cancelled) = oneshot::channel::<()>();

        self.fetch_vad_in_background(token.clone());
        tokio::spawn(run_download(manager, model, token, tx, cancelled));

        Ok(Box::pin(CancelOnDrop {
            progress: rx,
            _cancel: cancel,
        }))
    }

    /// Get the settings directory path using XDG-compliant directories
    fn get_settings_dir() -> Result<PathBuf, PlatformError> {
        directories::ProjectDirs::from("com", "purr", "purr")
            .map(|proj_dirs| proj_dirs.config_dir().to_path_buf())
            .ok_or_else(|| {
                PlatformError::settings_persistence(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Could not determine config directory",
                ))
            })
    }

    /// Get the full path to the settings file
    fn get_settings_file_path() -> Result<PathBuf, PlatformError> {
        let settings_dir = Self::get_settings_dir()?;
        Ok(settings_dir.join(SETTINGS_FILE))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_catalog::{
        entry::{ModelEntry, Precision},
        family::ModelGroup,
    };
    use purr_core::model::registry::Registry;
    use std::collections::HashSet;

    fn entries() -> Result<Vec<ModelEntry>, Box<dyn std::error::Error>> {
        let manager = ModelManager::with_registry(PathBuf::new(), Registry::embedded()?);
        Ok(manager
            .registry()
            .models()
            .iter()
            .map(|spec| ModelEntry::from_info(&model_info(&manager, spec, false)))
            .collect())
    }

    #[test]
    fn every_desktop_model_lands_in_a_known_group() -> Result<(), Box<dyn std::error::Error>> {
        let unknown: Vec<String> = entries()?
            .into_iter()
            .filter(|entry| entry.group == ModelGroup::Other)
            .map(|entry| entry.id)
            .collect();
        assert_eq!(unknown, Vec::<String>::new());
        Ok(())
    }

    #[test]
    fn desktop_model_labels_are_unique() -> Result<(), Box<dyn std::error::Error>> {
        let entries = entries()?;
        let labels: HashSet<&str> = entries.iter().map(|entry| entry.label.as_str()).collect();
        assert_eq!(labels.len(), entries.len());
        Ok(())
    }

    #[test]
    fn a_parse_error_names_the_line_but_not_its_text() {
        let contents = "version = 2\n\n[model]\nhugging_face_token = hf_secret\n";
        let error = toml::from_str::<Settings>(contents).unwrap_err();
        assert_eq!(
            describe_parse_error(contents, &error),
            "string values must be quoted, expected literal string (line 4)".to_owned()
        );
    }

    #[test]
    fn desktop_precisions_are_all_understood() -> Result<(), Box<dyn std::error::Error>> {
        let unknown: Vec<String> = entries()?
            .into_iter()
            .filter(|entry| matches!(entry.precision, Precision::Other(_)))
            .map(|entry| entry.id)
            .collect();
        assert_eq!(unknown, Vec::<String>::new());
        Ok(())
    }
}
