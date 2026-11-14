//! Loading a model on the first device of the plan that takes it.

use std::path::Path;

use purr_common::devices::DeviceOrder;
use tracing::{info, warn};

use crate::{
    accel::{
        inventory::Inventory,
        plan::{Step, resolve},
        report::{AccelAttempt, AccelReport, AttemptOutcome, UnavailableReason},
    },
    error::{AccelerationError, Result, TranscriptionError},
};

use super::{Engine, OpenSpec, decoder::WindowDecoder, error::OpenError};

/// A model loaded on a device, and the report of how that device was chosen.
pub struct Opened {
    pub decoder: Box<dyn WindowDecoder>,
    pub accel: AccelReport,
}

/// Loads `model` with `engine` on the first device of `order` that takes it.
pub fn open_on_best_device(
    engine: &dyn Engine,
    model: &Path,
    order: &DeviceOrder,
    threads: usize,
) -> Result<Opened> {
    std::fs::metadata(model)?;
    engine
        .availability()
        .map_err(|reason| TranscriptionError::EngineUnavailable {
            engine: engine.id(),
            reason,
        })?;
    let inventory = Inventory::detect(engine);
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
        let spec = OpenSpec {
            model,
            candidate: &candidate,
            devices: &inventory.devices,
            threads,
        };
        let outcome = match engine.open(&spec) {
            Ok(decoder) => {
                let accel = AccelReport {
                    selected: candidate.backend,
                    key: candidate.key,
                    device: candidate.backend.is_gpu().then(|| candidate.name.clone()),
                    attempts,
                };
                log_selection(&accel);
                return Ok(Opened { decoder, accel });
            }
            Err(OpenError::NoDevice) => AttemptOutcome::Unavailable(UnavailableReason::NoDevice),
            Err(OpenError::Device(failure)) => AttemptOutcome::Failed(failure),
            Err(error @ OpenError::Engine(_)) => {
                return Err(TranscriptionError::Open { source: error }.into());
            }
        };
        warn!(
            "{} acceleration failed on {}: {outcome}",
            candidate.backend.pretty_name(),
            candidate.name
        );
        attempts.push(AccelAttempt::new(candidate.key, outcome));
    }

    Err(AccelerationError::Exhausted { attempts }.into())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accel::engine_report::fakes::Scripted;
    use crate::error::Error;
    use purr_common::{accel::EngineUnavailable, model::engine::EngineId};
    use std::path::PathBuf;

    #[test]
    fn an_engine_that_cannot_be_used_is_reported_with_its_reason_before_any_device_is_tried() {
        let why = EngineUnavailable::LibraryMissing {
            library: "transcribe.dll".to_owned(),
            searched: vec![PathBuf::from("runtime")],
        };
        let engine = Scripted {
            id: EngineId::TranscribeCpp,
            outcome: Err(why.clone()),
            devices: Vec::new(),
            built: Vec::new(),
        };
        let model = std::env::current_exe().unwrap();
        let outcome = open_on_best_device(&engine, &model, &DeviceOrder::default(), 1);
        let Err(Error::Transcription {
            source: TranscriptionError::EngineUnavailable { engine, reason },
        }) = outcome
        else {
            panic!("expected the engine to be reported unavailable");
        };
        assert_eq!(engine, EngineId::TranscribeCpp);
        assert_eq!(reason, why);
    }
}
