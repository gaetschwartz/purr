//! Messages exchanged with the transcription worker (`js/worker.js`).
//!
//! Both directions carry JSON text so the shapes are defined once, here, and covered by native
//! tests. Audio samples travel beside the JSON as a transferred `Float32Array`.

use crate::gpu_check::GpuProbe;
use purr_common::hugging_face_token::HuggingFaceToken;
use purr_common::platform::TranscriptSegment;
use purr_common::web_model::attributes::OnnxDtype;
use purr_common::web_model::registry::WebModelSpec;
use serde::{Deserialize, Serialize};

/// Correlates a worker request with the events it produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RequestId(u64);

impl RequestId {
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }
}

/// Pinned model the worker hands to transformers.js.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerModel {
    pub repo: String,
    pub revision: String,
    pub encoder_dtype: OnnxDtype,
    pub decoder_dtype: OnnxDtype,
    pub requires_shader_f16: bool,
}

impl From<&WebModelSpec> for WorkerModel {
    fn from(spec: &WebModelSpec) -> Self {
        Self {
            repo: spec.repo.clone(),
            revision: spec.revision.clone(),
            encoder_dtype: spec.encoder_dtype,
            decoder_dtype: spec.decoder_dtype,
            requires_shader_f16: spec.requires_shader_f16(),
        }
    }
}

/// The ONNX Runtime execution provider a model runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provider {
    WebGpu,
    /// WebAssembly on the CPU.
    Cpu,
}

/// First message of a worker: where it finds the ONNX runtime files and the GPU check recording.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerSetup {
    Configure {
        ort_module_url: String,
        ort_wasm_url: String,
        canary_url: String,
    },
}

/// Requests sent to the worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerRequest {
    /// Download, cache and initialise a model without transcribing.
    LoadModel {
        request_id: RequestId,
        model: WorkerModel,
        provider: Provider,
        #[serde(default, deserialize_with = "HuggingFaceToken::deserialize_optional")]
        hugging_face_token: Option<HuggingFaceToken>,
    },
    /// Look for a WebGPU adapter.
    ProbeGpu { request_id: RequestId },
    /// Transcribe the check recording with `model` on WebGPU.
    CheckGpu {
        request_id: RequestId,
        model: WorkerModel,
        english_only: bool,
        #[serde(default, deserialize_with = "HuggingFaceToken::deserialize_optional")]
        hugging_face_token: Option<HuggingFaceToken>,
    },
    /// Transcribe the 16 kHz mono samples that accompany the message.
    Transcribe {
        request_id: RequestId,
        model: WorkerModel,
        provider: Provider,
        language: Option<String>,
        translate: bool,
        english_only: bool,
        #[serde(default, deserialize_with = "HuggingFaceToken::deserialize_optional")]
        hugging_face_token: Option<HuggingFaceToken>,
    },
}

impl WorkerRequest {
    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        match self {
            Self::LoadModel { request_id, .. }
            | Self::ProbeGpu { request_id }
            | Self::CheckGpu { request_id, .. }
            | Self::Transcribe { request_id, .. } => *request_id,
        }
    }
}

/// Why the worker gave up on a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureKind {
    WebGpuUnavailable,
    ShaderF16Unavailable,
    /// WebGPU failed while loading or running a model.
    WebGpuFailed,
    ModelLoad,
    Inference,
    Protocol,
    /// Raised on this side when the worker stops or cannot start.
    WorkerCrashed,
    /// Raised on this side when the worker sends something unparseable.
    MalformedEvent,
    /// Raised on this side when the worker is replaced while the request is queued.
    WorkerRestarted,
}

/// Events the worker reports back.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerEvent {
    /// Bytes of one model file received so far.
    ModelFile {
        request_id: RequestId,
        file: String,
        loaded: u64,
        total: u64,
    },
    ModelFileDone {
        request_id: RequestId,
        file: String,
    },
    ModelReady {
        request_id: RequestId,
    },
    /// The WebGPU adapter the worker found, `None` when there is none.
    GpuProbed {
        request_id: RequestId,
        probe: Option<GpuProbe>,
    },
    /// The text the GPU wrote for the check recording.
    GpuChecked {
        request_id: RequestId,
        text: String,
    },
    /// The model is loaded and speech decoding has started.
    Transcribing {
        request_id: RequestId,
    },
    /// The language Whisper recognised when none was requested.
    LanguageDetected {
        request_id: RequestId,
        language: String,
    },
    /// Live text of one segment; `text` is everything decoded for it so far.
    Segment {
        request_id: RequestId,
        chunk_index: usize,
        text: String,
        start_time: f64,
        end_time: f64,
    },
    /// Final, de-duplicated segment list.
    Segments {
        request_id: RequestId,
        segments: Vec<TranscriptSegment>,
    },
    Completed {
        request_id: RequestId,
        processing_time: f64,
        audio_duration: f64,
        word_count: usize,
    },
    Failed {
        request_id: Option<RequestId>,
        kind: FailureKind,
        message: String,
    },
}

impl WorkerEvent {
    /// The request the event belongs to, `None` for failures not tied to a request.
    #[must_use]
    pub const fn request_id(&self) -> Option<RequestId> {
        match self {
            Self::ModelFile { request_id, .. }
            | Self::ModelFileDone { request_id, .. }
            | Self::ModelReady { request_id }
            | Self::GpuProbed { request_id, .. }
            | Self::GpuChecked { request_id, .. }
            | Self::Transcribing { request_id }
            | Self::LanguageDetected { request_id, .. }
            | Self::Segment { request_id, .. }
            | Self::Segments { request_id, .. }
            | Self::Completed { request_id, .. } => Some(*request_id),
            Self::Failed { request_id, .. } => *request_id,
        }
    }
}

/// Which event ends the stream of a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    LoadModel,
    ProbeGpu,
    CheckGpu,
    Transcribe,
}

impl WorkerRequest {
    #[must_use]
    pub const fn kind(&self) -> RequestKind {
        match self {
            Self::LoadModel { .. } => RequestKind::LoadModel,
            Self::ProbeGpu { .. } => RequestKind::ProbeGpu,
            Self::CheckGpu { .. } => RequestKind::CheckGpu,
            Self::Transcribe { .. } => RequestKind::Transcribe,
        }
    }
}

impl WorkerEvent {
    /// Whether no further events follow for a request of the given kind.
    #[must_use]
    pub const fn ends(&self, kind: RequestKind) -> bool {
        match (self, kind) {
            (Self::Failed { .. }, _)
            | (Self::ModelReady { .. }, RequestKind::LoadModel)
            | (Self::GpuProbed { .. }, RequestKind::ProbeGpu)
            | (Self::GpuChecked { .. }, RequestKind::CheckGpu)
            | (Self::Completed { .. }, RequestKind::Transcribe) => true,
            (
                Self::ModelFile { .. }
                | Self::ModelFileDone { .. }
                | Self::ModelReady { .. }
                | Self::GpuProbed { .. }
                | Self::GpuChecked { .. }
                | Self::Transcribing { .. }
                | Self::LanguageDetected { .. }
                | Self::Segment { .. }
                | Self::Segments { .. }
                | Self::Completed { .. },
                _,
            ) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn model() -> WorkerModel {
        WorkerModel {
            repo: "onnx-community/whisper-tiny".to_owned(),
            revision: "ff4177021cc41f7db950912b73ea4fdf7d01d8e7".to_owned(),
            encoder_dtype: OnnxDtype::Fp32,
            decoder_dtype: OnnxDtype::Q4,
            requires_shader_f16: false,
        }
    }

    #[test]
    fn the_worker_model_carries_the_shader_f16_requirement()
    -> Result<(), Box<dyn std::error::Error>> {
        let registry = purr_common::web_model::registry::WebModelRegistry::embedded()?;
        assert!(!WorkerModel::from(registry.get("base-q4")?).requires_shader_f16);
        assert!(WorkerModel::from(registry.get("large-v3-turbo-q4f16")?).requires_shader_f16);
        Ok(())
    }

    #[test]
    fn transcribe_request_has_the_shape_worker_js_reads() -> Result<(), serde_json::Error> {
        let request = WorkerRequest::Transcribe {
            request_id: RequestId::new(7),
            model: model(),
            provider: Provider::Cpu,
            language: Some("fr".to_owned()),
            translate: false,
            english_only: false,
            hugging_face_token: None,
        };
        assert_eq!(
            serde_json::to_value(&request)?,
            json!({
                "type": "Transcribe",
                "request_id": 7,
                "model": {
                    "repo": "onnx-community/whisper-tiny",
                    "revision": "ff4177021cc41f7db950912b73ea4fdf7d01d8e7",
                    "encoder_dtype": "fp32",
                    "decoder_dtype": "q4",
                    "requires_shader_f16": false,
                },
                "provider": "Cpu",
                "language": "fr",
                "translate": false,
                "english_only": false,
                "hugging_face_token": null,
            })
        );
        Ok(())
    }

    #[test]
    fn the_token_travels_as_a_plain_field_of_the_request() -> Result<(), serde_json::Error> {
        let request = WorkerRequest::LoadModel {
            request_id: RequestId::new(3),
            model: model(),
            provider: Provider::Cpu,
            hugging_face_token: HuggingFaceToken::new("hf_dummy"),
        };
        assert_eq!(
            serde_json::to_value(&request)?["hugging_face_token"],
            json!("hf_dummy")
        );
        let decoded: WorkerRequest = serde_json::from_str(&serde_json::to_string(&request)?)?;
        assert_eq!(decoded, request);
        Ok(())
    }

    #[test]
    fn the_debug_form_of_a_request_hides_the_token() {
        let request = WorkerRequest::Transcribe {
            request_id: RequestId::new(1),
            model: model(),
            provider: Provider::WebGpu,
            language: None,
            translate: false,
            english_only: false,
            hugging_face_token: HuggingFaceToken::new("hf_dummy"),
        };
        assert_eq!(
            format!("{request:?}"),
            "Transcribe { request_id: RequestId(1), model: WorkerModel { \
             repo: \"onnx-community/whisper-tiny\", \
             revision: \"ff4177021cc41f7db950912b73ea4fdf7d01d8e7\", \
             encoder_dtype: Fp32, decoder_dtype: Q4, requires_shader_f16: false }, \
             provider: WebGpu, language: None, translate: false, english_only: false, \
             hugging_face_token: Some(HuggingFaceToken(<redacted>)) }"
        );
        let check = WorkerRequest::CheckGpu {
            request_id: RequestId::new(2),
            model: model(),
            english_only: false,
            hugging_face_token: HuggingFaceToken::new("hf_dummy"),
        };
        assert_eq!(
            format!("{check:?}"),
            "CheckGpu { request_id: RequestId(2), model: WorkerModel { \
             repo: \"onnx-community/whisper-tiny\", \
             revision: \"ff4177021cc41f7db950912b73ea4fdf7d01d8e7\", \
             encoder_dtype: Fp32, decoder_dtype: Q4, requires_shader_f16: false }, \
             english_only: false, hugging_face_token: Some(HuggingFaceToken(<redacted>)) }"
        );
    }

    #[test]
    fn automatic_language_serialises_as_null() -> Result<(), serde_json::Error> {
        let request = WorkerRequest::Transcribe {
            request_id: RequestId::new(1),
            model: model(),
            provider: Provider::WebGpu,
            language: None,
            translate: true,
            english_only: true,
            hugging_face_token: None,
        };
        assert_eq!(serde_json::to_value(&request)?["language"], json!(null));
        Ok(())
    }

    #[test]
    fn setup_message_has_the_shape_worker_js_reads() -> Result<(), serde_json::Error> {
        let setup = WorkerSetup::Configure {
            ort_module_url: "/assets/ort.mjs".to_owned(),
            ort_wasm_url: "/assets/ort.wasm".to_owned(),
            canary_url: "/assets/canary.pcm".to_owned(),
        };
        assert_eq!(
            serde_json::to_value(&setup)?,
            json!({
                "type": "Configure",
                "ort_module_url": "/assets/ort.mjs",
                "ort_wasm_url": "/assets/ort.wasm",
                "canary_url": "/assets/canary.pcm",
            })
        );
        Ok(())
    }

    #[test]
    fn load_model_and_gpu_requests_round_trip() -> Result<(), serde_json::Error> {
        for request in [
            WorkerRequest::LoadModel {
                request_id: RequestId::new(3),
                model: model(),
                provider: Provider::WebGpu,
                hugging_face_token: None,
            },
            WorkerRequest::ProbeGpu {
                request_id: RequestId::new(4),
            },
            WorkerRequest::CheckGpu {
                request_id: RequestId::new(5),
                model: model(),
                english_only: false,
                hugging_face_token: HuggingFaceToken::new("hf_dummy"),
            },
        ] {
            let decoded: WorkerRequest = serde_json::from_str(&serde_json::to_string(&request)?)?;
            assert_eq!(decoded, request);
        }
        Ok(())
    }

    #[test]
    fn parses_the_gpu_probe_and_check() -> Result<(), serde_json::Error> {
        let probed = parse(&json!({
            "type": "GpuProbed",
            "request_id": 4,
            "probe": {"fingerprint": "{\"vendor\":\"\"}", "shader_f16": true},
        }))?;
        assert_eq!(
            probed,
            WorkerEvent::GpuProbed {
                request_id: RequestId::new(4),
                probe: Some(GpuProbe {
                    fingerprint: "{\"vendor\":\"\"}".to_owned(),
                    shader_f16: true,
                }),
            }
        );
        assert!(probed.ends(RequestKind::ProbeGpu));
        let none = parse(&json!({"type": "GpuProbed", "request_id": 4, "probe": null}))?;
        assert_eq!(
            none,
            WorkerEvent::GpuProbed {
                request_id: RequestId::new(4),
                probe: None,
            }
        );
        let checked = parse(&json!({"type": "GpuChecked", "request_id": 5, "text": " >>"}))?;
        assert!(checked.ends(RequestKind::CheckGpu));
        assert!(!checked.ends(RequestKind::Transcribe));
        Ok(())
    }

    fn parse(value: &serde_json::Value) -> Result<WorkerEvent, serde_json::Error> {
        serde_json::from_str(&value.to_string())
    }

    #[test]
    fn parses_model_progress() -> Result<(), serde_json::Error> {
        let event = parse(&json!({
            "type": "ModelFile",
            "request_id": 2,
            "file": "onnx/encoder_model.onnx",
            "loaded": 1024,
            "total": 32_904_992,
        }))?;
        assert_eq!(
            event,
            WorkerEvent::ModelFile {
                request_id: RequestId::new(2),
                file: "onnx/encoder_model.onnx".to_owned(),
                loaded: 1024,
                total: 32_904_992,
            }
        );
        Ok(())
    }

    #[test]
    fn parses_live_segment_with_whole_number_times() -> Result<(), serde_json::Error> {
        let event = parse(&json!({
            "type": "Segment",
            "request_id": 2,
            "chunk_index": 0,
            "text": " Et un grand bravo",
            "start_time": 0,
            "end_time": 2.5,
        }))?;
        assert_eq!(
            event,
            WorkerEvent::Segment {
                request_id: RequestId::new(2),
                chunk_index: 0,
                text: " Et un grand bravo".to_owned(),
                start_time: 0.0,
                end_time: 2.5,
            }
        );
        Ok(())
    }

    #[test]
    fn parses_the_detected_language() -> Result<(), serde_json::Error> {
        let event = parse(&json!({
            "type": "LanguageDetected",
            "request_id": 4,
            "language": "fr",
        }))?;
        assert_eq!(
            event,
            WorkerEvent::LanguageDetected {
                request_id: RequestId::new(4),
                language: "fr".to_owned(),
            }
        );
        Ok(())
    }

    #[test]
    fn parses_final_segments_and_completion() -> Result<(), serde_json::Error> {
        let segments = parse(&json!({
            "type": "Segments",
            "request_id": 2,
            "segments": [{"text": " Bonjour", "start_time": 0.0, "end_time": 1.0}],
        }))?;
        assert_eq!(
            segments,
            WorkerEvent::Segments {
                request_id: RequestId::new(2),
                segments: vec![TranscriptSegment {
                    text: " Bonjour".to_owned(),
                    start_time: 0.0,
                    end_time: 1.0,
                }],
            }
        );
        let completed = parse(&json!({
            "type": "Completed",
            "request_id": 2,
            "processing_time": 1.5,
            "audio_duration": 4,
            "word_count": 12,
        }))?;
        assert_eq!(
            completed,
            WorkerEvent::Completed {
                request_id: RequestId::new(2),
                processing_time: 1.5,
                audio_duration: 4.0,
                word_count: 12,
            }
        );
        Ok(())
    }

    #[test]
    fn failure_without_request_parses_and_has_no_request_id() -> Result<(), serde_json::Error> {
        let event = parse(&json!({
            "type": "Failed",
            "request_id": null,
            "kind": "WebGpuUnavailable",
            "message": "WebGPU is not available in this browser",
        }))?;
        assert_eq!(event.request_id(), None);
        assert_eq!(
            event,
            WorkerEvent::Failed {
                request_id: None,
                kind: FailureKind::WebGpuUnavailable,
                message: "WebGPU is not available in this browser".to_owned(),
            }
        );
        Ok(())
    }

    #[test]
    fn unknown_event_type_is_rejected() {
        assert!(parse(&json!({"type": "Mystery", "request_id": 1})).is_err());
    }

    #[test]
    fn completion_ends_only_transcription_requests() {
        let completed = WorkerEvent::Completed {
            request_id: RequestId::new(1),
            processing_time: 0.0,
            audio_duration: 0.0,
            word_count: 0,
        };
        let ready = WorkerEvent::ModelReady {
            request_id: RequestId::new(1),
        };
        assert!(completed.ends(RequestKind::Transcribe));
        assert!(!completed.ends(RequestKind::LoadModel));
        assert!(ready.ends(RequestKind::LoadModel));
        assert!(!ready.ends(RequestKind::Transcribe));
    }
}
