use std::path::Path;

use tokio::task;
use tracing::{debug, info, warn};
use whisper_rs::{WhisperContext, WhisperContextParameters, WhisperState};

use crate::{
    ModelManager, TranscriptionConfig, WhisperError,
    accel::{
        config::{AccelConfig, AccelMode},
        coreml,
        inventory::Inventory,
        plan::{Candidate, Target, resolve},
        probe::probe_gpu,
        report::{AccelAttempt, AccelReport, AttemptFailure, AttemptOutcome, UnavailableReason},
    },
    error::{AccelerationError, Result},
};

pub(crate) struct LoadedModel {
    pub state: WhisperState,
    pub accel: AccelReport,
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

    let accel = config.accel.clone();
    task::spawn_blocking(move || load_with_acceleration(&model_path, &accel))
        .await
        .map_err(WhisperError::from)?
}

fn load_with_acceleration(model_path: &Path, config: &AccelConfig) -> Result<LoadedModel> {
    std::fs::metadata(model_path)?;
    let inventory = Inventory::detect();
    let plan = resolve(config, &inventory)?;
    let order = config.attempt_order();
    let rank = |backend| order.iter().position(|b| *b == backend);

    let mut attempts = plan.skipped;
    for candidate in &plan.candidates {
        match try_candidate(candidate, model_path) {
            Ok(state) => {
                attempts.retain(|a| rank(a.backend) < rank(candidate.backend));
                attempts.sort_by_key(|a| rank(a.backend));
                let report = AccelReport {
                    selected: candidate.backend,
                    device: match &candidate.target {
                        Target::Gpu(gpu) => Some(gpu.name.clone()),
                        Target::Cpu | Target::CoreMl => None,
                    },
                    attempts,
                };
                log_selection(&report, config);
                return Ok(LoadedModel {
                    state,
                    accel: report,
                });
            }
            Err(failure) => {
                warn!(
                    "{} acceleration failed: {failure}",
                    candidate.backend.pretty_name()
                );
                attempts.push(AccelAttempt {
                    backend: candidate.backend,
                    outcome: AttemptOutcome::Failed(failure),
                });
            }
        }
    }

    attempts.sort_by_key(|a| rank(a.backend));
    Err(AccelerationError::Exhausted { attempts }.into())
}

fn try_candidate(
    candidate: &Candidate,
    model_path: &Path,
) -> std::result::Result<WhisperState, AttemptFailure> {
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
    context.create_state().map_err(|error| {
        debug!("whisper state creation failed: {error}");
        AttemptFailure::StateInit
    })
}

fn log_selection(report: &AccelReport, config: &AccelConfig) {
    let forced = match config.mode {
        AccelMode::Auto => None,
        AccelMode::Force { backend, .. } => Some(backend),
    };
    for attempt in &report.attempts {
        let AttemptOutcome::Unavailable(reason) = &attempt.outcome else {
            continue;
        };
        let expected = Some(attempt.backend) != forced && *reason != UnavailableReason::NoDevice;
        if expected {
            debug!("{} not used: {reason}", attempt.backend.pretty_name());
        } else {
            warn!("{} not used: {reason}", attempt.backend.pretty_name());
        }
    }
    let device = report
        .device
        .as_deref()
        .map(|device| format!(" ({device})"))
        .unwrap_or_default();
    info!("Acceleration: {}{device}", report.selected.pretty_name());
}
