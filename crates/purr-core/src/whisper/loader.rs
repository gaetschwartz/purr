use std::path::Path;

use tokio::task;
use tracing::{debug, info, warn};
use whisper_rs::{WhisperContext, WhisperContextParameters, WhisperState};

use crate::{
    ModelManager, TranscriptionConfig, WhisperError,
    accel::{
        coreml,
        inventory::Inventory,
        plan::{Candidate, Step, Target, resolve},
        probe::probe_gpu,
        report::{AccelAttempt, AccelReport, AttemptFailure, AttemptOutcome},
    },
    error::{AccelerationError, Result},
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
        .map_err(WhisperError::from)?
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
        match try_candidate(&candidate, model_path) {
            Ok((state, tokens)) => {
                let report = AccelReport {
                    selected: candidate.backend,
                    key: candidate.key,
                    device: match &candidate.target {
                        Target::Gpu(gpu) => Some(gpu.name.clone()),
                        Target::Cpu | Target::CoreMl => None,
                    },
                    attempts,
                };
                log_selection(&report);
                return Ok(LoadedModel {
                    state,
                    tokens,
                    accel: report,
                });
            }
            Err(failure) => {
                warn!(
                    "{} acceleration failed on {}: {failure}",
                    candidate.backend.pretty_name(),
                    candidate.name
                );
                attempts.push(AccelAttempt::new(
                    candidate.key,
                    AttemptOutcome::Failed(failure),
                ));
            }
        }
    }

    Err(AccelerationError::Exhausted { attempts }.into())
}

fn try_candidate(
    candidate: &Candidate,
    model_path: &Path,
) -> std::result::Result<(WhisperState, SpecialTokens), AttemptFailure> {
    let mut params = WhisperContextParameters::default();
    match &candidate.target {
        Target::Cpu => {
            params.use_gpu(false);
        }
        Target::CoreMl => {
            let path = coreml::encoder_path(model_path);
            if !path.exists() {
                return Err(AttemptFailure::CoreMlModelMissing { path });
            }
            params.use_gpu(false);
        }
        Target::Gpu(gpu) => {
            probe_gpu(gpu)?;
            params.use_gpu(true).gpu_device(gpu.ordinal);
        }
    }

    let context = WhisperContext::new_with_params(model_path, params).map_err(|error| {
        debug!("whisper context creation failed: {error}");
        AttemptFailure::ContextInit
    })?;
    let tokens = SpecialTokens {
        eot: context.token_eot(),
    };
    let state = context.create_state().map_err(|error| {
        debug!("whisper state creation failed: {error}");
        AttemptFailure::StateInit
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
