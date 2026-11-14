use std::path::Path;

use tokio::task;
use tracing::{debug, info, warn};
use whisper_rs::{WhisperContext, WhisperContextParameters, WhisperState};

use crate::{
    Error, ModelManager, TranscriptionConfig,
    accel::{
        backend::AccelBackend,
        inventory::Inventory,
        plan::{Candidate, Step, resolve},
        report::{AccelAttempt, AccelReport, AttemptFailure, AttemptOutcome, UnavailableReason},
    },
    dev::Device,
    error::{AccelerationError, Result},
    whisper::{coreml, devices::locate, probe::probe_gpu},
};
use purr_common::devices::DeviceOrder;

pub(crate) struct LoadedModel {
    pub state: WhisperState,
    pub tokens: SpecialTokens,
    pub accel: AccelReport,
}

/// Token ids that mark where the model's text vocabulary ends.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SpecialTokens {
    /// End of text; every id from here on is a special or timestamp token
    pub eot: i32,
}

/// Load the Whisper model and create its state on the best acceleration backend that works.
pub(crate) async fn load_model(
    config: &TranscriptionConfig,
    model_manager: &ModelManager,
) -> Result<LoadedModel> {
    let model_path = if let Some(path) = &config.model_path {
        path.clone()
    } else {
        model_manager.find_default_model().await?
    };

    if config.verbose {
        info!("Loading Whisper model: {}", model_path.display());
    } else {
        info!("Loading model...");
    }

    let devices = config.devices.clone();
    task::spawn_blocking(move || load_with_acceleration(&model_path, &devices))
        .await
        .map_err(Error::from)?
}

fn load_with_acceleration(model_path: &Path, order: &DeviceOrder) -> Result<LoadedModel> {
    std::fs::metadata(model_path)?;
    let inventory = Inventory::detect();
    let plan = resolve(order, &inventory);

    let mut attempts = Vec::new();
    for step in plan.steps {
        let candidate = match step {
            Step::Skip(attempt) => {
                attempts.push(attempt);
                continue;
            }
            Step::Try(candidate) => candidate,
        };
        match try_candidate(&candidate, &inventory.devices, model_path) {
            Ok((state, tokens)) => {
                let report = AccelReport {
                    selected: candidate.backend,
                    key: candidate.key,
                    device: candidate.backend.is_gpu().then(|| candidate.name.clone()),
                    attempts,
                };
                log_selection(&report);
                return Ok(LoadedModel {
                    state,
                    tokens,
                    accel: report,
                });
            }
            Err(outcome) => {
                warn!(
                    "{} acceleration failed on {}: {outcome}",
                    candidate.backend.pretty_name(),
                    candidate.name
                );
                attempts.push(AccelAttempt::new(candidate.key, outcome));
            }
        }
    }

    Err(AccelerationError::Exhausted { attempts }.into())
}

fn try_candidate(
    candidate: &Candidate,
    devices: &[Device],
    model_path: &Path,
) -> std::result::Result<(WhisperState, SpecialTokens), AttemptOutcome> {
    let mut params = WhisperContextParameters::default();
    match candidate.backend {
        AccelBackend::Cpu => {
            params.use_gpu(false);
        }
        AccelBackend::CoreMl => {
            let path = coreml::encoder_path(model_path);
            if !path.exists() {
                return Err(AttemptOutcome::Failed(AttemptFailure::CoreMlModelMissing {
                    path,
                }));
            }
            params.use_gpu(false);
        }
        AccelBackend::Cuda
        | AccelBackend::Hip
        | AccelBackend::Metal
        | AccelBackend::Sycl
        | AccelBackend::Vulkan => {
            let gpu = locate(candidate, devices)
                .ok_or(AttemptOutcome::Unavailable(UnavailableReason::NoDevice))?;
            probe_gpu(&gpu).map_err(AttemptOutcome::Failed)?;
            params.use_gpu(true).gpu_device(gpu.ordinal);
        }
    }

    let context = WhisperContext::new_with_params(model_path, params).map_err(|error| {
        debug!("whisper context creation failed: {error}");
        AttemptOutcome::Failed(AttemptFailure::ContextInit)
    })?;
    let tokens = SpecialTokens {
        eot: context.token_eot(),
    };
    let state = context.create_state().map_err(|error| {
        debug!("whisper state creation failed: {error}");
        AttemptOutcome::Failed(AttemptFailure::StateInit)
    })?;
    Ok((state, tokens))
}

fn log_selection(report: &AccelReport) {
    for attempt in &report.attempts {
        let AttemptOutcome::Unavailable(reason) = &attempt.outcome else {
            continue;
        };
        warn!(
            "{} {} not used: {reason}",
            attempt.backend.pretty_name(),
            attempt.key.name
        );
    }
    let device = report
        .device
        .as_deref()
        .map(|device| format!(" ({device})"))
        .unwrap_or_default();
    info!("Acceleration: {}{device}", report.selected.pretty_name());
}
