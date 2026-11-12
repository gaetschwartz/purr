//! One transcription run, folded from the platform's status updates.

use super::transcript::Transcript;
use purr_common::{accel::LastRun, browser_engine::BrowserEngine, platform::TranscriptionStatus};

/// The step a running transcription is in, in the order they happen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    LoadingModel,
    DetectingLanguage,
    Transcribing,
}

/// Bytes of the model read so far, when the platform reports them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelLoad {
    pub loaded_bytes: u64,
    pub total_bytes: u64,
}

impl ModelLoad {
    /// The share loaded, only while bytes are actually arriving.
    #[must_use]
    pub fn fraction(self) -> Option<f64> {
        (self.loaded_bytes > 0 && self.loaded_bytes < self.total_bytes)
            .then(|| self.loaded_bytes as f64 / self.total_bytes as f64)
    }
}

/// Why a run stopped without a transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunFailure {
    pub message: String,
    /// What to try, when the platform knows.
    pub hint: Option<String>,
}

/// How a finished run went, as the platform measured it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Completion {
    pub processing_seconds: f64,
    pub audio_seconds: f64,
    pub word_count: usize,
}

/// Where a run stands.
#[derive(Debug, Clone, PartialEq)]
pub enum RunOutcome {
    Running,
    Completed(Completion),
    Failed(RunFailure),
}

/// A transcription run as the Transcribe screen shows it.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub stage: Stage,
    pub model_load: Option<ModelLoad>,
    /// What the engine ran on, once the model has loaded and the platform says.
    pub engine: Option<LastRun>,
    /// Where a browser run settled, once the platform says.
    pub browser_engine: Option<BrowserEngine>,
    pub transcript: Transcript,
    /// The Whisper code the platform recognised, when it was asked to detect the language.
    pub detected_language: Option<String>,
    pub outcome: RunOutcome,
    detect_language: bool,
    started_ms: f64,
    transcribing_since_ms: Option<f64>,
}

impl Run {
    /// A run that just started at `now_ms`; `detect_language` adds the detection stage.
    #[must_use]
    pub fn new(detect_language: bool, now_ms: f64) -> Self {
        Self {
            stage: Stage::LoadingModel,
            model_load: None,
            engine: None,
            browser_engine: None,
            transcript: Transcript::default(),
            detected_language: None,
            outcome: RunOutcome::Running,
            detect_language,
            started_ms: now_ms,
            transcribing_since_ms: None,
        }
    }

    /// Folds one status update in at `now_ms`.
    pub fn apply(&mut self, status: &TranscriptionStatus, now_ms: f64) {
        self.transcript.apply(status);
        match status {
            TranscriptionStatus::Starting | TranscriptionStatus::ProcessingAudio => {}
            TranscriptionStatus::LoadingModel {
                loaded_bytes,
                total_bytes,
            } => {
                self.model_load = Some(ModelLoad {
                    loaded_bytes: *loaded_bytes,
                    total_bytes: *total_bytes,
                });
            }
            TranscriptionStatus::Engine { run } => {
                self.engine = Some(run.clone());
            }
            TranscriptionStatus::BrowserEngine { engine } => {
                self.browser_engine = Some(*engine);
            }
            TranscriptionStatus::Transcribing => {
                self.transcribing_since_ms.get_or_insert(now_ms);
                self.advance(if self.detect_language {
                    Stage::DetectingLanguage
                } else {
                    Stage::Transcribing
                });
            }
            TranscriptionStatus::LanguageDetected { language } => {
                self.detected_language = Some(language.clone());
                self.transcribing_since_ms.get_or_insert(now_ms);
                self.advance(Stage::Transcribing);
            }
            TranscriptionStatus::InProgress { .. } | TranscriptionStatus::Segments { .. } => {
                self.transcribing_since_ms.get_or_insert(now_ms);
                self.advance(Stage::Transcribing);
            }
            TranscriptionStatus::Completed {
                processing_time,
                audio_duration,
                word_count,
            } => {
                self.outcome = RunOutcome::Completed(Completion {
                    processing_seconds: *processing_time,
                    audio_seconds: f64::from(*audio_duration),
                    word_count: *word_count,
                });
            }
            TranscriptionStatus::Error { error_message, .. } => {
                self.fail(error_message.clone(), None);
            }
            TranscriptionStatus::InitFailed {
                reason,
                error_details,
                ..
            } => self.fail(reason.clone(), error_details.clone()),
        }
    }

    /// Ends the run with `message`, unless it already ended.
    pub fn fail(&mut self, message: String, hint: Option<String>) {
        if self.outcome == RunOutcome::Running {
            self.outcome = RunOutcome::Failed(RunFailure { message, hint });
        }
    }

    fn advance(&mut self, stage: Stage) {
        self.stage = self.stage.max(stage);
    }

    /// When the run started, in the clock's milliseconds; it tells two runs of one file apart.
    #[must_use]
    pub const fn started_ms(&self) -> f64 {
        self.started_ms
    }

    /// Seconds since the run started.
    #[must_use]
    pub fn elapsed_seconds(&self, now_ms: f64) -> f64 {
        ((now_ms - self.started_ms) / 1000.0).max(0.0)
    }

    /// The audio covered so far: where the latest segment ends.
    #[must_use]
    pub fn transcribed_seconds(&self) -> f64 {
        self.transcript
            .segments()
            .iter()
            .map(|segment| segment.end_time.max(segment.start_time))
            .fold(0.0, f64::max)
    }

    /// The share of `audio_seconds` transcribed so far.
    #[must_use]
    pub fn fraction(&self, audio_seconds: f64) -> Option<f64> {
        if let RunOutcome::Completed(_) = self.outcome {
            return Some(1.0);
        }
        (audio_seconds > 0.0 && self.stage == Stage::Transcribing)
            .then(|| (self.transcribed_seconds() / audio_seconds).clamp(0.0, 1.0))
    }

    /// Seconds of audio per second of decoding since speech decoding began.
    #[must_use]
    pub fn live_speed(&self, now_ms: f64) -> Option<f64> {
        let since = self.transcribing_since_ms?;
        super::timing::speed(self.transcribed_seconds(), (now_ms - since) / 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(index: usize, end: f64) -> TranscriptionStatus {
        TranscriptionStatus::InProgress {
            chunk_index: index,
            text: format!("part {index}"),
            start_time: end - 4.0,
            end_time: end,
        }
    }

    #[test]
    fn detection_is_a_stage_only_when_asked_for() {
        let mut detecting = Run::new(true, 0.0);
        detecting.apply(&TranscriptionStatus::Transcribing, 1.0);
        assert_eq!(detecting.stage, Stage::DetectingLanguage);
        detecting.apply(
            &TranscriptionStatus::LanguageDetected {
                language: "fr".to_owned(),
            },
            2.0,
        );
        assert_eq!(detecting.stage, Stage::Transcribing);
        assert_eq!(detecting.detected_language.as_deref(), Some("fr"));

        let mut forced = Run::new(false, 0.0);
        forced.apply(&TranscriptionStatus::Transcribing, 1.0);
        assert_eq!(forced.stage, Stage::Transcribing);
    }

    #[test]
    fn the_engine_the_platform_reports_is_kept() {
        let mut run = Run::new(false, 0.0);
        assert_eq!(run.engine, None);
        let ran = LastRun {
            backend: purr_common::accel::AccelBackend::Vulkan,
            device: None,
            passed_over: Vec::new(),
        };
        run.apply(&TranscriptionStatus::Engine { run: ran.clone() }, 1.0);
        assert_eq!(run.engine, Some(ran));
        assert_eq!(run.stage, Stage::LoadingModel);
    }

    #[test]
    fn a_segment_moves_past_detection_even_without_a_language() {
        let mut run = Run::new(true, 0.0);
        run.apply(&TranscriptionStatus::Transcribing, 1.0);
        run.apply(&segment(0, 4.0), 2.0);
        assert_eq!(run.stage, Stage::Transcribing);
    }

    #[test]
    fn stages_never_go_back() {
        let mut run = Run::new(false, 0.0);
        run.apply(&segment(0, 4.0), 1.0);
        run.apply(
            &TranscriptionStatus::LoadingModel {
                loaded_bytes: 1,
                total_bytes: 2,
            },
            2.0,
        );
        assert_eq!(run.stage, Stage::Transcribing);
    }

    #[test]
    fn model_bytes_count_only_while_they_arrive() {
        let partial = ModelLoad {
            loaded_bytes: 50,
            total_bytes: 200,
        };
        assert_eq!(partial.fraction(), Some(0.25));
        let cached = ModelLoad {
            loaded_bytes: 200,
            total_bytes: 200,
        };
        assert_eq!(cached.fraction(), None);
        let not_started = ModelLoad {
            loaded_bytes: 0,
            total_bytes: 200,
        };
        assert_eq!(not_started.fraction(), None);
    }

    #[test]
    fn progress_and_speed_follow_the_segments() {
        let mut run = Run::new(false, 0.0);
        run.apply(&TranscriptionStatus::Transcribing, 1_000.0);
        run.apply(&segment(0, 8.0), 3_000.0);
        assert_eq!(run.fraction(40.0), Some(0.2));
        assert_eq!(run.live_speed(5_000.0), Some(2.0));
        assert_eq!(run.elapsed_seconds(5_000.0), 5.0);
    }

    #[test]
    fn completion_and_failure_end_the_run_once() {
        let mut run = Run::new(false, 0.0);
        run.apply(
            &TranscriptionStatus::Completed {
                processing_time: 4.0,
                audio_duration: 20.0,
                word_count: 12,
            },
            1.0,
        );
        run.fail("late".to_owned(), None);
        assert_eq!(
            run.outcome,
            RunOutcome::Completed(Completion {
                processing_seconds: 4.0,
                audio_seconds: 20.0,
                word_count: 12
            })
        );
        assert_eq!(run.fraction(20.0), Some(1.0));
    }

    #[test]
    fn platform_failures_keep_their_hint() {
        let mut run = Run::new(true, 0.0);
        run.apply(
            &TranscriptionStatus::InitFailed {
                component: "Model".to_owned(),
                reason: "Failed to fetch.".to_owned(),
                error_details: Some("Check your internet connection.".to_owned()),
            },
            1.0,
        );
        assert_eq!(
            run.outcome,
            RunOutcome::Failed(RunFailure {
                message: "Failed to fetch.".to_owned(),
                hint: Some("Check your internet connection.".to_owned()),
            })
        );
    }
}
