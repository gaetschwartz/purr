use crate::{
    model_catalog::state::ModelState,
    transcribe::{eta::EtaText, run::Stage, timing},
};
use std::fmt;

/// The words at the end of the audio card's second line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RunStatus {
    /// The model is downloading; the share once the size is known.
    Downloading(Option<f64>),
    CheckingDownload,
    LoadingModel,
    DetectingLanguage,
    /// Speech is being decoded; the time left once it can be estimated.
    Transcribing(Option<EtaText>),
    /// The run failed after transcribing this many seconds.
    Stopped(f64),
}

impl RunStatus {
    /// The status of a running run at `stage`, given the model's state and the time left.
    #[must_use]
    pub fn running(stage: Stage, model: &ModelState, eta: Option<EtaText>) -> Self {
        match (stage, model) {
            (Stage::LoadingModel, ModelState::Downloading(progress)) => {
                Self::Downloading(progress.fraction())
            }
            (Stage::LoadingModel, ModelState::Verifying) => Self::CheckingDownload,
            (
                Stage::LoadingModel,
                ModelState::NotDownloaded
                | ModelState::Downloaded
                | ModelState::Failed(_)
                | ModelState::NeedsRepair(_)
                | ModelState::Unavailable { .. },
            ) => Self::LoadingModel,
            (Stage::DetectingLanguage, _) => Self::DetectingLanguage,
            (Stage::Transcribing, _) => Self::Transcribing(eta),
        }
    }

    /// Whether the status is work in progress, drawn with a spinner.
    #[must_use]
    pub const fn busy(self) -> bool {
        match self {
            Self::Downloading(_)
            | Self::CheckingDownload
            | Self::LoadingModel
            | Self::DetectingLanguage => true,
            Self::Transcribing(_) | Self::Stopped(_) => false,
        }
    }
}

impl fmt::Display for RunStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Downloading(Some(done)) => {
                write!(f, "Downloading model {}%", (done * 100.0).round())
            }
            Self::Downloading(None) => f.write_str("Downloading model"),
            Self::CheckingDownload => f.write_str("Checking the download"),
            Self::LoadingModel => f.write_str("Loading model"),
            Self::DetectingLanguage => f.write_str("Detecting language"),
            Self::Transcribing(Some(eta)) => write!(f, "{eta}"),
            Self::Transcribing(None) => f.write_str("Transcribing"),
            Self::Stopped(reached) if *reached > 0.0 => {
                write!(f, "Stopped at {}", timing::clock(*reached))
            }
            Self::Stopped(_) => f.write_str("Stopped"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_download_shows_until_the_model_loads() {
        assert_eq!(
            RunStatus::running(Stage::LoadingModel, &ModelState::Verifying, None),
            RunStatus::CheckingDownload
        );
        assert_eq!(
            RunStatus::running(Stage::LoadingModel, &ModelState::Downloaded, None),
            RunStatus::LoadingModel
        );
        assert_eq!(
            RunStatus::running(
                Stage::Transcribing,
                &ModelState::Downloaded,
                Some(EtaText::Minutes(4))
            ),
            RunStatus::Transcribing(Some(EtaText::Minutes(4)))
        );
    }
}
