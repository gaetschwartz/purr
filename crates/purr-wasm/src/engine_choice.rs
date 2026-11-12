//! Picks the execution provider for a model in this browser, running the GPU check when no
//! recent verdict covers the adapter, browser, runtime and model.

use crate::gpu_check::{self, CanaryVerdict, CheckRecord, GpuProbe, ModelNeeds, Signature};
use crate::protocol::{FailureKind, Provider, WorkerEvent, WorkerModel, WorkerRequest};
use crate::worker_client::WorkerClient;
use futures::StreamExt as _;
use purr_common::accel::{AccelBackend, AccelPolicy};
use purr_common::browser_engine::{BrowserEngine, CpuReason, EnginePlan};
use purr_common::hugging_face_token::HuggingFaceToken;
use purr_common::web_model::registry::WebModelSpec;
use std::collections::HashMap;

const STORE_KEY: &str = "purr-gpu-checks";

/// Whether choosing may transcribe the check recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckMode {
    Run,
    Skip,
}

/// Where a model runs, and what a failure on the GPU is recorded against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice {
    pub engine: BrowserEngine,
    signature: Option<Signature>,
    cpu_can_run: bool,
}

impl Choice {
    #[must_use]
    pub const fn provider(&self) -> Provider {
        match self.engine {
            BrowserEngine::WebGpu => Provider::WebGpu,
            BrowserEngine::Cpu(_) => Provider::Cpu,
        }
    }

    /// The CPU, after the GPU failed with `kind`; `None` when the failure has nothing to do with the GPU.
    #[must_use]
    pub fn after_gpu_failure(&self, kind: FailureKind) -> Option<Self> {
        if self.engine != BrowserEngine::WebGpu || !self.cpu_can_run {
            return None;
        }
        Some(cpu(gpu_failure_reason(kind)?))
    }

    /// Keeps runs of this model on the CPU until the GPU is checked again.
    pub fn remember_gpu_failure(&self) {
        if let Some(signature) = &self.signature {
            remember(signature, CanaryVerdict::Failed);
        }
    }
}

/// The CPU reason a worker failure gives, for failures the CPU can recover from.
#[must_use]
pub const fn gpu_failure_reason(kind: FailureKind) -> Option<CpuReason> {
    match kind {
        FailureKind::WebGpuUnavailable => Some(CpuReason::NoWebGpu),
        FailureKind::ShaderF16Unavailable => Some(CpuReason::NoShaderF16),
        FailureKind::WebGpuFailed => Some(CpuReason::WebGpuFailed),
        FailureKind::ModelLoad
        | FailureKind::Inference
        | FailureKind::Protocol
        | FailureKind::WorkerCrashed
        | FailureKind::MalformedEvent
        | FailureKind::WorkerRestarted => None,
    }
}

fn local_storage() -> Option<web_sys::Storage> {
    match web_sys::window().map(|window| window.local_storage()) {
        Some(Ok(storage)) => storage,
        Some(Err(error)) => {
            tracing::warn!(
                "localStorage is unavailable, GPU checks are not kept: {}",
                crate::error::format_js_error(error)
            );
            None
        }
        None => None,
    }
}

fn records() -> HashMap<String, CheckRecord> {
    let Some(storage) = local_storage() else {
        return HashMap::new();
    };
    match storage.get_item(STORE_KEY) {
        Ok(Some(json)) => serde_json::from_str(&json).unwrap_or_else(|error| {
            tracing::warn!("The stored GPU checks are unreadable and will be redone: {error}");
            HashMap::new()
        }),
        Ok(None) => HashMap::new(),
        Err(error) => {
            tracing::warn!(
                "Could not read the stored GPU checks: {}",
                crate::error::format_js_error(error)
            );
            HashMap::new()
        }
    }
}

fn remember(signature: &Signature, verdict: CanaryVerdict) {
    let Some(storage) = local_storage() else {
        return;
    };
    let now_ms = js_sys::Date::now();
    let mut all = records();
    all.retain(|_, record| record.verdict_for(&record.signature, now_ms).is_some());
    all.insert(
        signature.as_str().to_owned(),
        CheckRecord {
            signature: signature.clone(),
            verdict,
            checked_at_ms: now_ms,
        },
    );
    match serde_json::to_string(&all) {
        Ok(json) => {
            if let Err(error) = storage.set_item(STORE_KEY, &json) {
                tracing::warn!(
                    "Could not keep the GPU check: {}",
                    crate::error::format_js_error(error)
                );
            }
        }
        Err(error) => tracing::warn!("Could not encode the GPU checks: {error}"),
    }
}

fn stored_for(signature: &Signature) -> Option<CheckRecord> {
    records().remove(signature.as_str())
}

async fn probe(worker: &WorkerClient) -> Result<Option<GpuProbe>, String> {
    let mut events = worker
        .submit(
            WorkerRequest::ProbeGpu {
                request_id: worker.next_request_id(),
            },
            None,
        )
        .map_err(|error| error.to_string())?;
    while let Some(event) = events.next().await {
        match event {
            WorkerEvent::GpuProbed { probe, .. } => return Ok(probe),
            WorkerEvent::Failed { message, .. } => return Err(message),
            other @ (WorkerEvent::ModelFile { .. }
            | WorkerEvent::ModelFileDone { .. }
            | WorkerEvent::ModelReady { .. }
            | WorkerEvent::GpuChecked { .. }
            | WorkerEvent::Transcribing { .. }
            | WorkerEvent::LanguageDetected { .. }
            | WorkerEvent::Segment { .. }
            | WorkerEvent::Segments { .. }
            | WorkerEvent::Completed { .. }) => {
                tracing::debug!("Ignoring {other:?} while probing the GPU");
            }
        }
    }
    Err("the worker stopped before reporting the GPU".to_owned())
}

/// The verdict of the check, or why it could not be given for reasons that have nothing to do
/// with the GPU, such as the run being cancelled.
async fn check(
    worker: &WorkerClient,
    spec: &WebModelSpec,
    hugging_face_token: Option<&HuggingFaceToken>,
    on_event: &mut impl FnMut(WorkerEvent),
) -> Result<CanaryVerdict, String> {
    let request = WorkerRequest::CheckGpu {
        request_id: worker.next_request_id(),
        model: WorkerModel::from(spec),
        english_only: spec.english_only(),
        hugging_face_token: hugging_face_token.cloned(),
    };
    let mut events = worker
        .submit(request, None)
        .map_err(|error| error.to_string())?;
    while let Some(event) = events.next().await {
        match event {
            WorkerEvent::GpuChecked { text, .. } => {
                let verdict = gpu_check::judge(&text);
                tracing::info!("GPU check of {} wrote {text:?}: {verdict:?}", spec.id);
                return Ok(verdict);
            }
            WorkerEvent::Failed { kind, message, .. } => {
                tracing::warn!("GPU check of {} failed ({kind:?}): {message}", spec.id);
                return match gpu_failure_reason(kind) {
                    Some(_) => Ok(CanaryVerdict::Failed),
                    None => Err(message),
                };
            }
            other @ (WorkerEvent::ModelFile { .. }
            | WorkerEvent::ModelFileDone { .. }
            | WorkerEvent::ModelReady { .. }
            | WorkerEvent::GpuProbed { .. }
            | WorkerEvent::Transcribing { .. }
            | WorkerEvent::LanguageDetected { .. }
            | WorkerEvent::Segment { .. }
            | WorkerEvent::Segments { .. }
            | WorkerEvent::Completed { .. }) => on_event(other),
        }
    }
    Err("the worker stopped before finishing the GPU check".to_owned())
}

const fn cpu(reason: CpuReason) -> Choice {
    Choice {
        engine: BrowserEngine::Cpu(reason),
        signature: None,
        cpu_can_run: true,
    }
}

fn needs(spec: &WebModelSpec) -> ModelNeeds<'_> {
    ModelNeeds {
        id: &spec.id,
        shader_f16: spec.requires_shader_f16(),
        runs_on_cpu: spec.runs_on_cpu(),
    }
}

/// The plan for `spec` under `policy`, and the signature its GPU verdict is kept under.
async fn plan(
    worker: &WorkerClient,
    spec: &WebModelSpec,
    policy: AccelPolicy,
) -> (EnginePlan, Option<Signature>) {
    let probe = match policy {
        AccelPolicy::Forced {
            backend: AccelBackend::Cpu,
            ..
        } => None,
        AccelPolicy::Auto | AccelPolicy::Forced { .. } => match probe(worker).await {
            Ok(probe) => probe,
            Err(error) => {
                tracing::warn!("Could not look for a WebGPU adapter: {error}");
                let reason = CpuReason::WebGpuFailed;
                let planned = if spec.runs_on_cpu() {
                    EnginePlan::Cpu(reason)
                } else {
                    EnginePlan::CannotRun(reason)
                };
                return (planned, None);
            }
        },
    };
    let signature = probe.as_ref().map(|probe| Signature::new(probe, &spec.id));
    let record = signature.as_ref().and_then(stored_for);
    let planned = gpu_check::plan(
        policy,
        probe.as_ref(),
        needs(spec),
        record.as_ref(),
        js_sys::Date::now(),
    );
    (planned, signature)
}

/// Where the next run of `spec` under `policy` goes, without running the GPU check.
pub async fn preview(
    worker: &WorkerClient,
    spec: &WebModelSpec,
    policy: AccelPolicy,
) -> EnginePlan {
    plan(worker, spec, policy).await.0
}

/// Where `spec` runs under `policy`, or why it cannot run in this browser. With
/// [`CheckMode::Run`] a missing or stale verdict is settled by transcribing the check recording,
/// downloading the model with `hugging_face_token`; the download and load events go to `on_event`.
pub async fn choose(
    worker: &WorkerClient,
    spec: &WebModelSpec,
    policy: AccelPolicy,
    mode: CheckMode,
    hugging_face_token: Option<&HuggingFaceToken>,
    on_event: &mut impl FnMut(WorkerEvent),
) -> Result<Choice, CpuReason> {
    let (planned, signature) = plan(worker, spec, policy).await;
    let gpu = Choice {
        engine: BrowserEngine::WebGpu,
        signature: signature.clone(),
        cpu_can_run: spec.runs_on_cpu(),
    };
    let after_check = |reason: CpuReason| {
        worker.release_idle();
        if spec.runs_on_cpu() {
            Ok(cpu(reason))
        } else {
            Err(reason)
        }
    };
    match (planned, mode, signature) {
        (EnginePlan::Cpu(reason), _, _) => Ok(cpu(reason)),
        (EnginePlan::CannotRun(reason), _, _) => Err(reason),
        (EnginePlan::WebGpu, _, _)
        | (EnginePlan::WebGpuAfterCheck, CheckMode::Skip, _)
        | (EnginePlan::WebGpuAfterCheck, CheckMode::Run, None) => Ok(gpu),
        (EnginePlan::WebGpuAfterCheck, CheckMode::Run, Some(signature)) => {
            match check(worker, spec, hugging_face_token, on_event).await {
                Ok(verdict) => {
                    remember(&signature, verdict);
                    match verdict {
                        CanaryVerdict::Correct => Ok(gpu),
                        CanaryVerdict::Wrong => after_check(CpuReason::WrongOutput),
                        CanaryVerdict::Failed => after_check(CpuReason::WebGpuFailed),
                    }
                }
                Err(reason) => {
                    tracing::warn!("The GPU check of {} did not finish: {reason}", spec.id);
                    after_check(CpuReason::WebGpuFailed)
                }
            }
        }
    }
}
