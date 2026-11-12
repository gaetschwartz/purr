//! Whether Transcribe can start, and what has to happen to the model first.

use crate::model_catalog::{
    copy,
    state::{ModelState, UnavailableReason},
};

/// The chosen file, as far as starting a run cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileReadiness {
    Missing,
    Undecodable,
    Usable,
}

/// The selected model, as far as starting a run cares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelChoice {
    /// The model list has not loaded yet.
    Loading,
    /// The list could not be loaded, or nothing in it can be used.
    Missing,
    Chosen(ModelState),
}

/// Why Transcribe is disabled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Blocker {
    NoFile,
    UndecodableFile,
    ModelsLoading,
    NoModel,
    /// The browser offers no WebGPU, which every browser model needs.
    NoWebGpu,
    CannotRun(UnavailableReason),
    /// The model's files are being checked.
    Checking,
    /// Another model is downloading; only one download runs at a time.
    OtherDownload,
}

/// What pressing Transcribe does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartPlan {
    Start(Preparation),
    Blocked(Blocker),
}

/// What the model needs before the run can use it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preparation {
    /// It is on this device.
    None,
    /// Download it; `replace` deletes damaged files first.
    Download { replace: bool },
    /// It is downloading already; wait for that download.
    JoinDownload,
}

/// One short sentence on why Transcribe cannot start.
#[must_use]
pub const fn blocked_text(blocker: &Blocker) -> &'static str {
    match blocker {
        Blocker::NoFile => "Choose a file first.",
        Blocker::UndecodableFile => "This file can't be decoded.",
        Blocker::ModelsLoading => "The model list is still loading.",
        Blocker::NoModel => "No model is available.",
        Blocker::NoWebGpu => "This browser has no WebGPU.",
        Blocker::CannotRun(reason) => copy::unavailable(*reason).1,
        Blocker::Checking => "The model's files are being checked.",
        Blocker::OtherDownload => "Wait for the other model download to finish.",
    }
}

/// What the model in `state` needs before a run, or why it cannot run.
///
/// `other_download` says whether a different model is downloading.
pub const fn prepare(state: &ModelState, other_download: bool) -> Result<Preparation, Blocker> {
    match state {
        ModelState::Downloaded => Ok(Preparation::None),
        ModelState::Downloading(_) => Ok(Preparation::JoinDownload),
        ModelState::Verifying => Err(Blocker::Checking),
        ModelState::Unavailable { reason, .. } => Err(Blocker::CannotRun(*reason)),
        ModelState::NotDownloaded | ModelState::Failed(_) if other_download => {
            Err(Blocker::OtherDownload)
        }
        ModelState::NeedsRepair(_) if other_download => Err(Blocker::OtherDownload),
        ModelState::NotDownloaded | ModelState::Failed(_) => {
            Ok(Preparation::Download { replace: false })
        }
        ModelState::NeedsRepair(_) => Ok(Preparation::Download { replace: true }),
    }
}

/// Decides what the Transcribe button does.
///
/// `no_webgpu` is true only in a browser that was found to lack WebGPU.
#[must_use]
pub fn plan_start(
    file: FileReadiness,
    model: &ModelChoice,
    no_webgpu: bool,
    other_download: bool,
) -> StartPlan {
    if no_webgpu {
        return StartPlan::Blocked(Blocker::NoWebGpu);
    }
    let preparation = match model {
        ModelChoice::Loading => Err(Blocker::ModelsLoading),
        ModelChoice::Missing => Err(Blocker::NoModel),
        ModelChoice::Chosen(state) => prepare(state, other_download),
    };
    match (file, preparation) {
        (_, Err(blocker)) => StartPlan::Blocked(blocker),
        (FileReadiness::Missing, Ok(_)) => StartPlan::Blocked(Blocker::NoFile),
        (FileReadiness::Undecodable, Ok(_)) => StartPlan::Blocked(Blocker::UndecodableFile),
        (FileReadiness::Usable, Ok(preparation)) => StartPlan::Start(preparation),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_catalog::activity::{Damage, DownloadFailure, DownloadProgress};
    use purr_common::platform::{IntegrityProblem, ModelFailureKind};

    fn chosen(state: ModelState) -> ModelChoice {
        ModelChoice::Chosen(state)
    }

    fn usable(model: &ModelChoice) -> StartPlan {
        plan_start(FileReadiness::Usable, model, false, false)
    }

    #[test]
    fn a_downloaded_model_starts_right_away() {
        assert_eq!(
            usable(&chosen(ModelState::Downloaded)),
            StartPlan::Start(Preparation::None)
        );
    }

    #[test]
    fn a_model_that_is_not_downloaded_is_downloaded_first() {
        assert_eq!(
            usable(&chosen(ModelState::NotDownloaded)),
            StartPlan::Start(Preparation::Download { replace: false })
        );
    }

    #[test]
    fn a_failed_download_is_tried_again_and_damaged_files_are_replaced() {
        let failed = ModelState::Failed(DownloadFailure {
            kind: ModelFailureKind::Network,
            message: "offline".to_owned(),
        });
        assert_eq!(
            usable(&chosen(failed)),
            StartPlan::Start(Preparation::Download { replace: false })
        );
        let damaged = ModelState::NeedsRepair(Damage {
            problem: IntegrityProblem::ChecksumMismatch,
            file: "model.bin".to_owned(),
        });
        assert_eq!(
            usable(&chosen(damaged)),
            StartPlan::Start(Preparation::Download { replace: true })
        );
    }

    #[test]
    fn a_running_download_of_the_model_is_joined() {
        assert_eq!(
            usable(&chosen(
                ModelState::Downloading(DownloadProgress::default())
            )),
            StartPlan::Start(Preparation::JoinDownload)
        );
    }

    #[test]
    fn another_download_blocks_only_models_that_need_one() {
        let model = chosen(ModelState::NotDownloaded);
        assert_eq!(
            plan_start(FileReadiness::Usable, &model, false, true),
            StartPlan::Blocked(Blocker::OtherDownload)
        );
        assert_eq!(
            plan_start(
                FileReadiness::Usable,
                &chosen(ModelState::Downloaded),
                false,
                true
            ),
            StartPlan::Start(Preparation::None)
        );
    }

    #[test]
    fn a_model_that_cannot_run_here_blocks_before_anything_downloads() {
        let state = ModelState::Unavailable {
            reason: UnavailableReason::NeedsShaderF16,
            installed: false,
        };
        assert_eq!(
            usable(&chosen(state)),
            StartPlan::Blocked(Blocker::CannotRun(UnavailableReason::NeedsShaderF16))
        );
        assert_eq!(
            usable(&chosen(ModelState::Verifying)),
            StartPlan::Blocked(Blocker::Checking)
        );
    }

    #[test]
    fn a_browser_without_webgpu_blocks_everything() {
        assert_eq!(
            plan_start(
                FileReadiness::Usable,
                &chosen(ModelState::Downloaded),
                true,
                false
            ),
            StartPlan::Blocked(Blocker::NoWebGpu)
        );
    }

    #[test]
    fn the_file_must_be_there_and_decodable() {
        let model = chosen(ModelState::NotDownloaded);
        assert_eq!(
            plan_start(FileReadiness::Missing, &model, false, false),
            StartPlan::Blocked(Blocker::NoFile)
        );
        assert_eq!(
            plan_start(FileReadiness::Undecodable, &model, false, false),
            StartPlan::Blocked(Blocker::UndecodableFile)
        );
    }

    #[test]
    fn a_model_problem_is_reported_before_a_missing_file() {
        assert_eq!(
            plan_start(FileReadiness::Missing, &ModelChoice::Loading, false, false),
            StartPlan::Blocked(Blocker::ModelsLoading)
        );
        assert_eq!(
            plan_start(FileReadiness::Missing, &ModelChoice::Missing, false, false),
            StartPlan::Blocked(Blocker::NoModel)
        );
    }
}
