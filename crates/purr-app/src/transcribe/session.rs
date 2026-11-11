//! The Transcribe flow's state, kept at the root so a run and its result outlive the screen.

use super::{
    chosen_file::{ChosenFile, FileProblem, OverviewState, WAVEFORM_BARS},
    language_choice::LanguageChoice,
    run::{Completion, Run, RunOutcome},
    timing,
};
use crate::{
    components::toast::{Toast, ToastKind, Toasts, use_toasts},
    history::{
        entry_builder::{self, Finished},
        save_state::SaveState,
        store::{HistoryStore, use_history},
    },
    hooks::use_models::{ModelsHandle, use_models},
    model_catalog::entry::{Language, ModelEntry},
    platform::{self, IS_WEB},
    settings_store::{SettingsStore, use_settings},
    utils::{
        clock::{now_ms, unix_ms},
        file_kind::may_contain_audio,
    },
};
use dioxus::{
    core::{Runtime, ScopeId, Task, spawn_forever},
    html::FileData,
    prelude::*,
};
use futures::StreamExt as _;
use purr_common::{
    history::entry::{HistoryEntry, HistoryId, LanguageOrigin},
    platform::{
        EngineOptions, FileId, FileSource, ModelId, Platform as _, PlatformError,
        TranscriptionRequest, TranscriptionStatus,
    },
};
use std::{collections::HashMap, path::PathBuf, rc::Rc};

/// A run with what it was started on.
#[derive(Debug, Clone, PartialEq)]
pub struct RunRecord {
    pub file: ChosenFile,
    pub model: ModelEntry,
    /// The language that applied, after the English-only lock.
    pub language: LanguageChoice,
    pub run: Run,
}

impl RunRecord {
    /// The spoken language to show: the detected one, else the one asked for.
    #[must_use]
    pub fn language_code(&self) -> Option<String> {
        self.run
            .detected_language
            .clone()
            .or_else(|| self.language.request_code())
    }
}

/// Handle to the Transcribe state, from [`use_transcribe_session`]. Cheap to copy; every method
/// may be called from any scope, including overlays and toast actions.
#[derive(Clone, Copy, PartialEq)]
pub struct TranscribeSession {
    file: Signal<Option<ChosenFile>>,
    reading: Signal<Option<String>>,
    problem: Signal<Option<FileProblem>>,
    overviews: Signal<HashMap<FileId, OverviewState>>,
    record: Signal<Option<RunRecord>>,
    task: Signal<Option<Task>>,
    saved: Signal<SaveState>,
    originals: Signal<HashMap<String, PathBuf>>,
    toasts: Toasts,
    models: ModelsHandle,
    history: HistoryStore,
    settings: SettingsStore,
}

impl TranscribeSession {
    /// The file picked for the next run (subscribes the caller).
    #[must_use]
    pub fn file(&self) -> Option<ChosenFile> {
        self.file.read().clone()
    }

    /// The name of the file being read, while it is.
    #[must_use]
    pub fn reading(&self) -> Option<String> {
        self.reading.read().clone()
    }

    /// Why the last pick failed, until dismissed.
    #[must_use]
    pub fn problem(&self) -> Option<FileProblem> {
        self.problem.read().clone()
    }

    /// The current or last run (subscribes the caller).
    #[must_use]
    pub fn record(&self) -> Option<RunRecord> {
        self.record.read().clone()
    }

    /// Whether a run is going on.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.record
            .read()
            .as_ref()
            .is_some_and(|record| record.run.outcome == RunOutcome::Running)
    }

    /// The length and waveform of `id`.
    #[must_use]
    pub fn overview(&self, id: &FileId) -> OverviewState {
        self.overviews
            .read()
            .get(id)
            .cloned()
            .unwrap_or(OverviewState::Reading)
    }

    /// Whether the last finished run is saved to History (subscribes the caller).
    #[must_use]
    pub fn save_state(&self) -> SaveState {
        self.saved.read().clone()
    }

    /// Lets the desktop webview play the original file of a saved transcription as `id`.
    pub fn serve_original(mut self, id: &HistoryId, path: PathBuf) {
        self.originals.write().insert(id.to_string(), path);
    }

    /// The original file registered under `id`, without subscribing.
    #[must_use]
    pub fn original_path(&self, id: &str) -> Option<PathBuf> {
        self.originals.peek().get(id).cloned()
    }

    /// The chosen or transcribed file with the id `id`, without subscribing.
    #[must_use]
    pub fn file_by_id(&self, id: &str) -> Option<ChosenFile> {
        let chosen = self.file.peek().clone().filter(|file| &*file.id == id);
        chosen.or_else(|| {
            self.record
                .peek()
                .as_ref()
                .map(|record| record.file.clone())
                .filter(|file| &*file.id == id)
        })
    }

    /// Forgets the last failed pick.
    pub fn dismiss_problem(mut self) {
        self.problem.set(None);
    }

    /// Reads, checks and stores `data` as the file for the next run.
    pub fn choose(self, data: FileData) {
        spawn_forever(async move { self.accept(data).await });
    }

    async fn accept(mut self, data: FileData) {
        let name = data.name();
        self.problem.set(None);
        let content_type = data.content_type().unwrap_or_default();
        if !may_contain_audio(&content_type, &name) {
            tracing::info!("Refused {name}: {content_type}");
            self.problem
                .set(Some(FileProblem::Unsupported { name, content_type }));
            return;
        }
        if let Some(limit) = platform::get_platform().max_upload_bytes()
            && data.size() > limit
        {
            self.problem.set(Some(FileProblem::TooLarge {
                size: data.size(),
                name,
                limit,
            }));
            return;
        }
        self.reading.set(Some(name.clone()));
        let stored = self.store(&data, &name, &content_type).await;
        self.reading.set(None);
        match stored {
            Ok(file) => {
                let previous = self.file.peek().clone();
                self.file.set(Some(file.clone()));
                if let Some(previous) = previous {
                    self.release(previous);
                }
                self.inspect(file);
            }
            Err(problem) => {
                tracing::warn!("Could not use {name}: {}", problem.message());
                self.problem.set(Some(problem));
            }
        }
    }

    async fn store(
        &self,
        data: &FileData,
        name: &str,
        content_type: &str,
    ) -> Result<ChosenFile, FileProblem> {
        if data.size() == 0 {
            return Err(FileProblem::Empty {
                name: name.to_owned(),
            });
        }
        let bytes = data
            .read_bytes()
            .await
            .map_err(|error| FileProblem::Unreadable {
                name: name.to_owned(),
                detail: error.to_string(),
            })?;
        if bytes.is_empty() {
            return Err(FileProblem::Empty {
                name: name.to_owned(),
            });
        }
        let id = platform::get_platform()
            .process_file(bytes.clone(), name.as_ref())
            .await
            .map_err(|error| FileProblem::Rejected {
                name: name.to_owned(),
                detail: error.to_string(),
            })?;
        tracing::info!("Stored {name} ({} bytes) as {id}", bytes.len());
        let source_path = (!IS_WEB).then(|| data.path()).filter(|path| path.is_file());
        Ok(ChosenFile {
            id,
            name: name.to_owned(),
            size: bytes.len() as u64,
            content_type: Some(content_type.to_owned()).filter(|kind| !kind.is_empty()),
            bytes,
            source_path,
        })
    }

    fn inspect(mut self, file: ChosenFile) {
        self.overviews
            .write()
            .insert(file.id.clone(), OverviewState::Reading);
        spawn_forever(async move {
            let overview = match platform::get_platform()
                .audio_overview(&file.id, WAVEFORM_BARS)
                .await
            {
                Ok(measured) => OverviewState::Ready {
                    duration_seconds: measured.duration_seconds,
                    peaks: Rc::new(measured.peaks),
                },
                Err(error) => {
                    tracing::warn!("Could not decode {}: {error}", file.name);
                    OverviewState::Undecodable(
                        if let PlatformError::AudioProcessing { source } = &error {
                            source.to_string()
                        } else {
                            error.to_string()
                        },
                    )
                }
            };
            if self.overviews.peek().contains_key(&file.id) {
                self.overviews.write().insert(file.id, overview);
            }
        });
    }

    /// Drops the chosen file, keeping it only if the last run still shows it.
    pub fn clear_file(mut self) {
        if let Some(previous) = self.file.take() {
            self.release(previous);
        }
    }

    /// Deletes the stored copy of `file` unless the chosen file or the last run still uses it.
    fn release(mut self, file: ChosenFile) {
        let chosen = self
            .file
            .peek()
            .as_ref()
            .is_some_and(|kept| kept.id == file.id);
        let shown = self
            .record
            .peek()
            .as_ref()
            .is_some_and(|record| record.file.id == file.id);
        if chosen || shown {
            return;
        }
        self.overviews.write().remove(&file.id);
        let toasts = self.toasts;
        spawn_forever(async move {
            if let Err(error) = platform::get_platform().cleanup(&file.id).await {
                tracing::error!("Could not remove the stored copy of {}: {error}", file.name);
                toasts.warning(
                    "A temporary copy was left behind",
                    format!("{} could not be removed: {error}", file.name),
                );
            }
        });
    }

    /// Transcribes the chosen file with `model`; the English-only lock applies to `language`.
    pub fn start(self, model: ModelEntry, language: LanguageChoice) {
        let Some(file) = self.file.peek().clone() else {
            self.toasts.error(
                "Choose a file first",
                "There is no audio to transcribe yet.",
            );
            return;
        };
        self.launch(file, model, language);
    }

    /// Runs the file of the last run again with `model`.
    pub fn run_again(self, model: ModelEntry) {
        let Some(previous) = self.record.peek().clone() else {
            return;
        };
        self.launch(previous.file, model, previous.language);
    }

    /// Tries the last run again with the same file, model and language.
    pub fn retry(self) {
        let Some(previous) = self.record.peek().clone() else {
            return;
        };
        self.launch(previous.file, previous.model, previous.language);
    }

    fn launch(mut self, file: ChosenFile, model: ModelEntry, language: LanguageChoice) {
        self.stop_task();
        let language = language.for_model(model.language == Language::EnglishOnly);
        let settings = self.settings.peek();
        let request = TranscriptionRequest {
            file: FileSource::Uploaded(file.id.clone()),
            model: ModelId::new(model.id.clone()),
            language: language.request_code(),
            translate: settings.model.translate,
            engine: EngineOptions::from_settings(&settings),
        };
        let run = Run::new(language == LanguageChoice::Automatic, now_ms());
        tracing::info!(
            "Transcribing {} with {} ({:?})",
            file.name,
            model.id,
            request.language
        );
        let previous = self.record.peek().clone();
        self.saved.set(SaveState::Idle);
        self.record.set(Some(RunRecord {
            file: file.clone(),
            model,
            language,
            run,
        }));
        if let Some(previous) = previous
            && previous.file.id != file.id
        {
            self.release(previous.file);
        }
        let task = spawn_forever(async move { self.drive(request).await });
        self.task.set(Some(task));
    }

    async fn drive(mut self, request: TranscriptionRequest) {
        let mut stream = match platform::get_platform().transcribe(request).await {
            Ok(stream) => stream,
            Err(error) => {
                tracing::error!("Could not start the transcription: {error}");
                self.fail(describe(&error));
                return;
            }
        };
        while let Some(update) = stream.next().await {
            match update {
                Ok(status) => {
                    let ended = matches!(
                        status,
                        TranscriptionStatus::Completed { .. }
                            | TranscriptionStatus::Error { .. }
                            | TranscriptionStatus::InitFailed { .. }
                    );
                    if let Some(record) = self.record.write().as_mut() {
                        record.run.apply(&status, now_ms());
                    }
                    if ended {
                        break;
                    }
                }
                Err(error) => {
                    tracing::error!("The transcription failed: {error}");
                    self.fail(describe(&error));
                    break;
                }
            }
        }
        self.fail("The transcription stopped before it finished.".to_owned());
        self.task.set(None);
        self.announce_end();
    }

    fn fail(mut self, message: String) {
        if let Some(record) = self.record.write().as_mut() {
            record.run.fail(message, None);
        }
    }

    fn announce_end(self) {
        let Some(record) = self.record.peek().clone() else {
            return;
        };
        match record.run.outcome {
            RunOutcome::Completed(completion) => {
                self.save_to_history(&record, completion);
                self.toasts.success(
                    "Transcription finished",
                    format!(
                        "{} of audio in {}",
                        timing::clock(completion.audio_seconds),
                        timing::took(completion.processing_seconds)
                    ),
                );
            }
            RunOutcome::Failed(_) => {
                self.toasts.error("Transcription failed", record.file.name);
            }
            RunOutcome::Running => {}
        }
    }

    fn save_to_history(mut self, record: &RunRecord, completion: Completion) {
        let overview = self.overviews.peek().get(&record.file.id).cloned();
        let (duration, peaks) = match overview {
            Some(OverviewState::Ready {
                duration_seconds,
                peaks,
            }) => (duration_seconds, peaks.as_ref().clone()),
            Some(OverviewState::Reading | OverviewState::Undecodable(_)) | None => {
                (completion.audio_seconds, Vec::new())
            }
        };
        let id = match entry_builder::new_id() {
            Ok(id) => id,
            Err(error) => {
                self.saved.set(SaveState::Idle);
                tracing::error!("Could not name the history entry: {error}");
                self.toasts
                    .error("Couldn't save to History", error.to_string());
                return;
            }
        };
        let origin = if record.run.detected_language.is_some() {
            LanguageOrigin::Detected
        } else {
            LanguageOrigin::Chosen
        };
        let entry = entry_builder::build(Finished {
            id,
            created_at_ms: unix_ms(),
            file_name: record.file.name.clone(),
            source_path: record.file.source_path.clone(),
            model_id: record.model.id.clone(),
            model_label: record.model.label.clone(),
            language: record.language_code(),
            language_origin: origin,
            processing_seconds: completion.processing_seconds,
            audio_seconds: duration,
            segments: record.run.transcript.segments(),
            peaks,
        });
        self.write_to_history(Rc::new(entry));
    }

    fn write_to_history(mut self, entry: Rc<HistoryEntry>) {
        self.saved.set(SaveState::Saving);
        spawn_forever(async move {
            let outcome = match self.history.save(&entry).await {
                Ok(saved) if saved.evicted.contains(&entry.id) => Err(
                    "History is full of newer transcriptions, so this one was dropped.".to_owned(),
                ),
                Ok(_) => Ok(()),
                Err(error) => {
                    tracing::error!("Could not save {} to History: {error}", entry.file_name);
                    Err(error.to_string())
                }
            };
            if *self.saved.peek() != SaveState::Saving {
                return;
            }
            match outcome {
                Ok(()) => self.saved.set(SaveState::Saved(entry.id.clone())),
                Err(message) => {
                    self.toasts
                        .error("Couldn't save to History", message.clone());
                    self.saved.set(SaveState::Failed { message, entry });
                }
            }
        });
    }

    /// Writes the last finished run to History again after a failed save.
    pub fn retry_save(self) {
        let state = self.saved.peek().clone();
        if let SaveState::Failed { entry, .. } = state {
            self.write_to_history(entry);
        }
    }

    fn stop_task(mut self) {
        if let Some(task) = self.task.take() {
            task.cancel();
        }
    }

    /// Stops the run and goes back to the chosen file; nothing is kept.
    pub fn cancel(mut self) {
        self.stop_task();
        let cancelled_record = self.record.take();
        if let Some(record) = &cancelled_record
            && self
                .file
                .peek()
                .as_ref()
                .is_none_or(|file| file.id != record.file.id)
        {
            self.file.set(Some(record.file.clone()));
        }
        let detail = if IS_WEB && self.models.is_downloading() {
            "Nothing was saved and the file is still selected. Cancelling restarts the browser's transcription worker, which also stopped the model download; start it again from Models."
        } else if IS_WEB {
            "Nothing was saved and the file is still selected. Cancelling restarts the browser's transcription worker, so the next run loads the model again."
        } else {
            "Nothing was saved and the file is still selected."
        };
        let mut toast = Toast::new(ToastKind::Info, "Transcription cancelled").with_detail(detail);
        if let Some(record) = cancelled_record {
            toast = toast.with_action("Start again", move || {
                if self.file_by_id(&record.file.id).is_some() {
                    self.launch(
                        record.file.clone(),
                        record.model.clone(),
                        record.language.clone(),
                    );
                }
            });
        }
        self.toasts.show(toast);
    }

    /// Leaves the result and the chosen file for a new pick.
    pub fn new_transcription(mut self) {
        self.stop_task();
        let record = self.record.take();
        let file = self.file.take();
        if let Some(record) = record {
            self.release(record.file);
        }
        if let Some(file) = file {
            self.release(file);
        }
    }
}

fn describe(error: &PlatformError) -> String {
    if let PlatformError::ModelNotFound { model_id } = error {
        format!("The model {model_id} is not on this device. Download it or choose another one.")
    } else {
        error.to_string()
    }
}

/// Creates the session at the app root; call once, after the toast provider.
pub fn use_transcribe_session_provider() -> TranscribeSession {
    let toasts = use_toasts();
    let models = use_models();
    let history = use_history();
    let settings = use_settings();
    let session = use_hook(|| {
        Runtime::current().in_scope(ScopeId::ROOT, || TranscribeSession {
            file: Signal::new(None),
            reading: Signal::new(None),
            problem: Signal::new(None),
            overviews: Signal::new(HashMap::new()),
            record: Signal::new(None),
            task: Signal::new(None),
            saved: Signal::new(SaveState::Idle),
            originals: Signal::new(HashMap::new()),
            toasts,
            models,
            history,
            settings,
        })
    });
    use_context_provider(|| session)
}

/// The session provided at the root.
pub fn use_transcribe_session() -> TranscribeSession {
    use_context::<TranscribeSession>()
}
