use super::{
    activity::{Damage, DownloadFailure, DownloadProgress, ModelActivity},
    entry::ModelEntry,
};
use crate::hooks::use_models::GpuSupport;

/// Why a model cannot run on this device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnavailableReason {
    /// The model needs the WebGPU `shader-f16` feature, which this GPU lacks.
    NeedsShaderF16,
    /// The model needs WebGPU and the browser offers none.
    NeedsWebGpu,
}

/// Where a model stands, for display and for deciding which actions apply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelState {
    NotDownloaded,
    Downloaded,
    Downloading(DownloadProgress),
    Verifying,
    Failed(DownloadFailure),
    /// A check found the stored files wrong; downloading again repairs them.
    NeedsRepair(Damage),
    /// The model cannot run here; it may still be stored.
    Unavailable {
        reason: UnavailableReason,
        installed: bool,
    },
}

impl ModelState {
    /// Whether the model's files are stored on this device and usable.
    #[must_use]
    pub const fn is_on_device(&self) -> bool {
        match self {
            Self::Downloaded
            | Self::Unavailable {
                installed: true, ..
            } => true,
            Self::NotDownloaded
            | Self::Downloading(_)
            | Self::Verifying
            | Self::Failed(_)
            | Self::NeedsRepair(_)
            | Self::Unavailable {
                installed: false, ..
            } => false,
        }
    }

    /// The download progress, while one runs.
    #[must_use]
    pub const fn progress(&self) -> Option<&DownloadProgress> {
        match self {
            Self::Downloading(progress) => Some(progress),
            Self::NotDownloaded
            | Self::Downloaded
            | Self::Verifying
            | Self::Failed(_)
            | Self::NeedsRepair(_)
            | Self::Unavailable { .. } => None,
        }
    }

    /// Whether a download, a check or a failure is the thing to show for the model.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        match self {
            Self::Downloading(_) | Self::Verifying | Self::Failed(_) => true,
            Self::NotDownloaded
            | Self::Downloaded
            | Self::NeedsRepair(_)
            | Self::Unavailable { .. } => false,
        }
    }
}

/// Why this GPU cannot run `entry`, if it cannot. While the GPU is still being detected nothing is ruled out.
#[must_use]
pub fn unavailable_reason(entry: &ModelEntry, gpu: &GpuSupport) -> Option<UnavailableReason> {
    if !entry.requires_shader_f16 {
        return None;
    }
    match gpu {
        GpuSupport::Checking => None,
        GpuSupport::Unavailable => Some(UnavailableReason::NeedsWebGpu),
        GpuSupport::Available { shader_f16, .. } => {
            (!shader_f16).then_some(UnavailableReason::NeedsShaderF16)
        }
    }
}

/// The state of `entry`: what is happening to it wins over what is stored.
#[must_use]
pub fn model_state(
    entry: &ModelEntry,
    activity: Option<&ModelActivity>,
    gpu: &GpuSupport,
) -> ModelState {
    match activity {
        Some(ModelActivity::Downloading(progress)) => return ModelState::Downloading(*progress),
        Some(ModelActivity::Verifying) => return ModelState::Verifying,
        Some(ModelActivity::Failed(failure)) => return ModelState::Failed(failure.clone()),
        Some(ModelActivity::Damaged(damage)) => return ModelState::NeedsRepair(damage.clone()),
        None => {}
    }
    if let Some(reason) = unavailable_reason(entry, gpu) {
        return ModelState::Unavailable {
            reason,
            installed: entry.installed,
        };
    }
    if entry.installed {
        ModelState::Downloaded
    } else {
        ModelState::NotDownloaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::platform::{
        IntegrityProblem, ModelFailureKind, ModelInfo, ModelMetadata, model_flag,
    };

    fn entry(installed: bool, needs_f16: bool) -> ModelEntry {
        let metadata = ModelMetadata::new(
            "onnx".to_owned(),
            "whisper".to_owned(),
            "multilingual".to_owned(),
            "1".to_owned(),
        )
        .with_platform_attribute("family".to_owned(), "tiny".to_owned())
        .with_platform_attribute(
            model_flag::REQUIRES_SHADER_F16.to_owned(),
            needs_f16.to_string(),
        );
        ModelEntry::from_info(&ModelInfo::new(
            "tiny".to_owned(),
            "tiny".to_owned(),
            String::new(),
            1,
            installed,
            metadata,
        ))
    }

    const NO_F16: GpuSupport = GpuSupport::Available {
        name: String::new(),
        shader_f16: false,
    };

    #[test]
    fn stored_files_decide_when_nothing_is_happening() {
        let gpu = GpuSupport::Checking;
        assert_eq!(
            model_state(&entry(true, false), None, &gpu),
            ModelState::Downloaded
        );
        assert_eq!(
            model_state(&entry(false, false), None, &gpu),
            ModelState::NotDownloaded
        );
    }

    #[test]
    fn activity_wins_over_stored_files() {
        let progress = DownloadProgress {
            loaded_bytes: 5,
            ..DownloadProgress::default()
        };
        assert_eq!(
            model_state(
                &entry(true, false),
                Some(&ModelActivity::Downloading(progress)),
                &GpuSupport::Checking
            ),
            ModelState::Downloading(progress)
        );
        let damage = Damage {
            problem: IntegrityProblem::ChecksumMismatch,
            file: "a".to_owned(),
        };
        assert_eq!(
            model_state(
                &entry(false, false),
                Some(&ModelActivity::Damaged(damage.clone())),
                &GpuSupport::Checking
            ),
            ModelState::NeedsRepair(damage)
        );
        let failure = DownloadFailure {
            kind: ModelFailureKind::Network,
            message: "offline".to_owned(),
        };
        assert_eq!(
            model_state(
                &entry(false, false),
                Some(&ModelActivity::Failed(failure.clone())),
                &GpuSupport::Checking
            ),
            ModelState::Failed(failure)
        );
        assert_eq!(
            model_state(
                &entry(true, false),
                Some(&ModelActivity::Verifying),
                &GpuSupport::Checking
            ),
            ModelState::Verifying
        );
    }

    #[test]
    fn a_missing_gpu_feature_makes_only_flagged_models_unavailable() {
        assert_eq!(
            model_state(&entry(false, true), None, &NO_F16),
            ModelState::Unavailable {
                reason: UnavailableReason::NeedsShaderF16,
                installed: false
            }
        );
        assert_eq!(
            model_state(&entry(false, false), None, &NO_F16),
            ModelState::NotDownloaded
        );
    }

    #[test]
    fn no_webgpu_at_all_blocks_flagged_models_and_detection_blocks_nothing() {
        assert_eq!(
            unavailable_reason(&entry(false, true), &GpuSupport::Unavailable),
            Some(UnavailableReason::NeedsWebGpu)
        );
        assert_eq!(
            unavailable_reason(&entry(false, true), &GpuSupport::Checking),
            None
        );
        let capable = GpuSupport::Available {
            name: String::new(),
            shader_f16: true,
        };
        assert_eq!(unavailable_reason(&entry(false, true), &capable), None);
    }

    #[test]
    fn an_unavailable_model_can_still_be_stored() {
        let state = model_state(&entry(true, true), None, &NO_F16);
        assert!(state.is_on_device());
        assert!(!ModelState::NotDownloaded.is_on_device());
    }
}
