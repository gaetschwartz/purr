//! Translates worker events into the platform's status and progress types.

use crate::model_progress::ModelProgress;
use crate::protocol::{FailureKind, WorkerEvent};
use purr_common::platform::{ModelOperationProgress, TranscriptionStatus};
use purr_common::web_model::registry::WebModelSpec;

/// `message` as a sentence, so a hint can follow it.
fn as_sentence(message: &str) -> String {
    let trimmed = message.trim();
    match trimmed.chars().last() {
        None | Some('.' | '!' | '?' | '…') => trimmed.to_owned(),
        Some(_) => format!("{trimmed}."),
    }
}

fn failure_status(kind: FailureKind, message: String) -> TranscriptionStatus {
    let message = as_sentence(&message);
    let init_failed = |component: &str, hint: &str| TranscriptionStatus::InitFailed {
        component: component.to_owned(),
        reason: message.clone(),
        error_details: Some(hint.to_owned()),
    };
    match kind {
        FailureKind::WebGpuUnavailable => init_failed(
            "WebGPU",
            "This model needs WebGPU. Choose a smaller one; it can run on the CPU.",
        ),
        FailureKind::ShaderF16Unavailable => init_failed(
            "WebGPU",
            "Choose a smaller model; it does not need this GPU feature.",
        ),
        FailureKind::WebGpuFailed => {
            init_failed("WebGPU", "Choose a smaller model; it can run on the CPU.")
        }
        FailureKind::ModelLoad => init_failed(
            "Model",
            "Check your internet connection and try again. Files that were already downloaded are kept.",
        ),
        FailureKind::WorkerCrashed | FailureKind::MalformedEvent | FailureKind::Protocol => {
            init_failed("Worker", "Reload the page and try again.")
        }
        FailureKind::Inference | FailureKind::WorkerRestarted => TranscriptionStatus::Error {
            context: "Transcription".to_owned(),
            error_message: message,
        },
    }
}

/// Maps the events of one transcription request to statuses.
pub struct TranscriptionMapper {
    progress: ModelProgress,
    cached: bool,
}

impl TranscriptionMapper {
    #[must_use]
    pub fn new(spec: &WebModelSpec, cached: bool) -> Self {
        Self {
            progress: ModelProgress::for_model(spec),
            cached,
        }
    }

    /// Reading a cached model is reported as fully downloaded so the UI does not claim a download.
    fn loading(&self) -> TranscriptionStatus {
        let total_bytes = self.progress.total_bytes();
        TranscriptionStatus::LoadingModel {
            loaded_bytes: if self.cached {
                total_bytes
            } else {
                self.progress.loaded_bytes()
            },
            total_bytes,
        }
    }

    /// The status to report for `event`, if it changes anything the user sees.
    pub fn map(&mut self, event: WorkerEvent) -> Option<TranscriptionStatus> {
        match event {
            WorkerEvent::ModelFile { file, loaded, .. } => {
                self.progress.record(&file, loaded);
                Some(self.loading())
            }
            WorkerEvent::ModelFileDone { file, .. } => {
                self.progress.finish(&file);
                Some(self.loading())
            }
            WorkerEvent::ModelReady { .. }
            | WorkerEvent::GpuProbed { .. }
            | WorkerEvent::GpuChecked { .. } => None,
            WorkerEvent::Transcribing { .. } => Some(TranscriptionStatus::Transcribing),
            WorkerEvent::LanguageDetected { language, .. } => {
                Some(TranscriptionStatus::LanguageDetected { language })
            }
            WorkerEvent::Segment {
                chunk_index,
                text,
                start_time,
                end_time,
                ..
            } => Some(TranscriptionStatus::InProgress {
                chunk_index,
                text,
                start_time,
                end_time,
                words: Vec::new(),
            }),
            WorkerEvent::Segments { segments, .. } => {
                Some(TranscriptionStatus::Segments { segments })
            }
            WorkerEvent::Completed {
                processing_time,
                audio_duration,
                word_count,
                ..
            } => Some(TranscriptionStatus::Completed {
                processing_time,
                audio_duration: audio_duration as f32,
                word_count,
            }),
            WorkerEvent::Failed { kind, message, .. } => Some(failure_status(kind, message)),
        }
    }
}

/// What a model preload reports after an event.
#[derive(Debug, PartialEq, Eq)]
pub enum PreloadStep {
    Progress(ModelOperationProgress),
    Loaded,
    Failed(String),
    Ignored,
}

/// Maps the events of one model preload to progress updates.
pub struct PreloadMapper {
    model_id: String,
    progress: ModelProgress,
    started_ms: f64,
}

impl PreloadMapper {
    #[must_use]
    pub fn new(spec: &WebModelSpec, started_ms: f64) -> Self {
        Self {
            model_id: spec.id.clone(),
            progress: ModelProgress::for_model(spec),
            started_ms,
        }
    }

    fn downloading(&self, now_ms: f64) -> ModelOperationProgress {
        let bytes_downloaded = self.progress.loaded_bytes();
        let seconds = (now_ms - self.started_ms) / 1000.0;
        let speed_bps = (seconds > 0.0).then(|| (bytes_downloaded as f64 / seconds) as u64);
        ModelOperationProgress::Downloading {
            model_id: self.model_id.clone(),
            bytes_downloaded,
            total_bytes: Some(self.progress.total_bytes()),
            speed_bps,
        }
    }

    pub fn map(&mut self, event: WorkerEvent, now_ms: f64) -> PreloadStep {
        match event {
            WorkerEvent::ModelFile { file, loaded, .. } => {
                self.progress.record(&file, loaded);
                PreloadStep::Progress(self.downloading(now_ms))
            }
            WorkerEvent::ModelFileDone { file, .. } => {
                self.progress.finish(&file);
                PreloadStep::Progress(self.downloading(now_ms))
            }
            WorkerEvent::ModelReady { .. } => PreloadStep::Loaded,
            WorkerEvent::Failed { message, .. } => PreloadStep::Failed(message),
            WorkerEvent::GpuProbed { .. }
            | WorkerEvent::GpuChecked { .. }
            | WorkerEvent::Transcribing { .. }
            | WorkerEvent::LanguageDetected { .. }
            | WorkerEvent::Segment { .. }
            | WorkerEvent::Segments { .. }
            | WorkerEvent::Completed { .. } => PreloadStep::Ignored,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::RequestId;
    use purr_common::platform::TranscriptSegment;
    use purr_common::web_model::registry::WebModelRegistry;

    const ID: RequestId = RequestId::new(1);

    fn tiny() -> Result<WebModelSpec, Box<dyn std::error::Error>> {
        Ok(WebModelRegistry::embedded()?.get("tiny")?.clone())
    }

    #[test]
    fn model_bytes_become_loading_status() -> Result<(), Box<dyn std::error::Error>> {
        let spec = tiny()?;
        let mut mapper = TranscriptionMapper::new(&spec, false);
        let status = mapper.map(WorkerEvent::ModelFile {
            request_id: ID,
            file: "onnx/encoder_model.onnx".to_owned(),
            loaded: 10,
            total: 32_904_992,
        });
        assert_eq!(
            status,
            Some(TranscriptionStatus::LoadingModel {
                loaded_bytes: 10,
                total_bytes: spec.total_size_bytes(),
            })
        );
        Ok(())
    }

    #[test]
    fn cached_model_is_never_reported_as_downloading() -> Result<(), Box<dyn std::error::Error>> {
        let spec = tiny()?;
        let mut mapper = TranscriptionMapper::new(&spec, true);
        let status = mapper.map(WorkerEvent::ModelFile {
            request_id: ID,
            file: "onnx/encoder_model.onnx".to_owned(),
            loaded: 10,
            total: 32_904_992,
        });
        assert_eq!(
            status,
            Some(TranscriptionStatus::LoadingModel {
                loaded_bytes: spec.total_size_bytes(),
                total_bytes: spec.total_size_bytes(),
            })
        );
        Ok(())
    }

    #[test]
    fn segments_and_completion_map_one_to_one() -> Result<(), Box<dyn std::error::Error>> {
        let mut mapper = TranscriptionMapper::new(&tiny()?, false);
        let segment = TranscriptSegment {
            text: " Bonjour".to_owned(),
            start_time: 0.0,
            end_time: 1.0,
            words: Vec::new(),
        };
        assert_eq!(
            mapper.map(WorkerEvent::Segments {
                request_id: ID,
                segments: vec![segment.clone()],
            }),
            Some(TranscriptionStatus::Segments {
                segments: vec![segment]
            })
        );
        assert_eq!(
            mapper.map(WorkerEvent::Completed {
                request_id: ID,
                processing_time: 2.0,
                audio_duration: 4.0,
                word_count: 1,
            }),
            Some(TranscriptionStatus::Completed {
                processing_time: 2.0,
                audio_duration: 4.0,
                word_count: 1,
            })
        );
        Ok(())
    }

    #[test]
    fn ready_model_reports_nothing_until_transcribing() -> Result<(), Box<dyn std::error::Error>> {
        let mut mapper = TranscriptionMapper::new(&tiny()?, false);
        assert_eq!(mapper.map(WorkerEvent::ModelReady { request_id: ID }), None);
        assert_eq!(
            mapper.map(WorkerEvent::Transcribing { request_id: ID }),
            Some(TranscriptionStatus::Transcribing)
        );
        Ok(())
    }

    #[test]
    fn detected_language_is_reported() -> Result<(), Box<dyn std::error::Error>> {
        let mut mapper = TranscriptionMapper::new(&tiny()?, false);
        assert_eq!(
            mapper.map(WorkerEvent::LanguageDetected {
                request_id: ID,
                language: "fr".to_owned(),
            }),
            Some(TranscriptionStatus::LanguageDetected {
                language: "fr".to_owned()
            })
        );
        Ok(())
    }

    #[test]
    fn missing_webgpu_is_an_initialisation_failure() -> Result<(), Box<dyn std::error::Error>> {
        let mut mapper = TranscriptionMapper::new(&tiny()?, false);
        let status = mapper.map(WorkerEvent::Failed {
            request_id: Some(ID),
            kind: FailureKind::WebGpuUnavailable,
            message: "no gpu".to_owned(),
        });
        assert_eq!(
            status,
            Some(TranscriptionStatus::InitFailed {
                component: "WebGPU".to_owned(),
                reason: "no gpu.".to_owned(),
                error_details: Some(
                    "This model needs WebGPU. Choose a smaller one; it can run on the CPU."
                        .to_owned()
                ),
            })
        );
        Ok(())
    }

    #[test]
    fn a_webgpu_failure_that_reaches_the_user_points_at_a_smaller_model()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut mapper = TranscriptionMapper::new(&tiny()?, false);
        let status = mapper.map(WorkerEvent::Failed {
            request_id: Some(ID),
            kind: FailureKind::WebGpuFailed,
            message: "WebGPU failed: device lost".to_owned(),
        });
        assert_eq!(
            status,
            Some(TranscriptionStatus::InitFailed {
                component: "WebGPU".to_owned(),
                reason: "WebGPU failed: device lost.".to_owned(),
                error_details: Some("Choose a smaller model; it can run on the CPU.".to_owned()),
            })
        );
        Ok(())
    }

    #[test]
    fn inference_failure_is_a_transcription_error() -> Result<(), Box<dyn std::error::Error>> {
        let mut mapper = TranscriptionMapper::new(&tiny()?, false);
        let status = mapper.map(WorkerEvent::Failed {
            request_id: Some(ID),
            kind: FailureKind::Inference,
            message: "out of memory".to_owned(),
        });
        assert_eq!(
            status,
            Some(TranscriptionStatus::Error {
                context: "Transcription".to_owned(),
                error_message: "out of memory.".to_owned(),
            })
        );
        Ok(())
    }

    #[test]
    fn a_failure_reason_ends_as_a_sentence_before_its_hint() {
        assert_eq!(as_sentence("Failed to fetch"), "Failed to fetch.");
        assert_eq!(as_sentence("Done. "), "Done.");
        assert_eq!(as_sentence("Why?"), "Why?");
        assert_eq!(as_sentence(""), "");
    }

    #[test]
    fn preload_reports_download_speed() -> Result<(), Box<dyn std::error::Error>> {
        let spec = tiny()?;
        let mut mapper = PreloadMapper::new(&spec, 10_000.0);
        let step = mapper.map(
            WorkerEvent::ModelFile {
                request_id: ID,
                file: "config.json".to_owned(),
                loaded: 2_000,
                total: 2_243,
            },
            12_000.0,
        );
        assert_eq!(
            step,
            PreloadStep::Progress(ModelOperationProgress::Downloading {
                model_id: "tiny".to_owned(),
                bytes_downloaded: 2_000,
                total_bytes: Some(spec.total_size_bytes()),
                speed_bps: Some(1_000),
            })
        );
        Ok(())
    }

    #[test]
    fn preload_ends_when_the_model_is_ready() -> Result<(), Box<dyn std::error::Error>> {
        let spec = tiny()?;
        let mut mapper = PreloadMapper::new(&spec, 0.0);
        assert_eq!(
            mapper.map(WorkerEvent::ModelReady { request_id: ID }, 0.0),
            PreloadStep::Loaded
        );
        Ok(())
    }
}
