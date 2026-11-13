use crate::{
    model_catalog::{
        activity::{DownloadProgress, VerifyProgress},
        format::{check_percent, percent, rate_and_eta},
        state::ModelState,
    },
    transcribe::{eta::EtaText, run::Stage, timing},
};
use std::fmt;

/// The words at the end of the audio card's second line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RunStatus {
    /// The model is downloading, as far as it has come.
    Downloading(DownloadProgress),
    /// The downloaded files are being hashed.
    CheckingDownload(VerifyProgress),
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
                Self::Downloading(*progress)
            }
            (Stage::LoadingModel, ModelState::Verifying(progress)) => {
                Self::CheckingDownload(*progress)
            }
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
            | Self::CheckingDownload(_)
            | Self::LoadingModel
            | Self::DetectingLanguage => true,
            Self::Transcribing(_) | Self::Stopped(_) => false,
        }
    }
}

impl fmt::Display for RunStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Downloading(progress) => {
                f.write_str("Downloading model")?;
                if let Some(done) = percent(progress) {
                    write!(f, " {done}%")?;
                }
                let pace = rate_and_eta(progress);
                if !pace.is_empty() {
                    write!(f, " · {pace}")?;
                }
                Ok(())
            }
            Self::CheckingDownload(progress) => {
                f.write_str("Checking the download")?;
                if let Some(done) = check_percent(progress) {
                    write!(f, " {done}%")?;
                }
                Ok(())
            }
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

    fn download(loaded_bytes: u64, bytes_per_second: Option<u64>) -> DownloadProgress {
        DownloadProgress {
            loaded_bytes,
            total_bytes: Some(1_000_000),
            bytes_per_second,
        }
    }

    #[test]
    fn a_download_shows_until_the_model_loads() {
        let checking = VerifyProgress {
            checked_bytes: 1,
            total_bytes: Some(4),
        };
        assert_eq!(
            RunStatus::running(Stage::LoadingModel, &ModelState::Verifying(checking), None),
            RunStatus::CheckingDownload(checking)
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

    #[test]
    fn a_download_names_its_speed_and_time_left_once_known() {
        assert_eq!(
            RunStatus::Downloading(download(450_000, Some(1_048_576))).to_string(),
            "Downloading model 45% · 1.0 MiB/s · 1 s left"
        );
    }

    #[test]
    fn a_download_leaves_out_what_is_not_known_yet() {
        assert_eq!(
            RunStatus::Downloading(download(450_000, None)).to_string(),
            "Downloading model 45%"
        );
        assert_eq!(
            RunStatus::Downloading(DownloadProgress::default()).to_string(),
            "Downloading model"
        );
        let unknown_total = DownloadProgress {
            loaded_bytes: 10,
            total_bytes: None,
            bytes_per_second: Some(2_048),
        };
        assert_eq!(
            RunStatus::Downloading(unknown_total).to_string(),
            "Downloading model · 2.0 KiB/s"
        );
    }

    #[test]
    fn a_check_shows_its_share_once_known() {
        let half = VerifyProgress {
            checked_bytes: 2,
            total_bytes: Some(4),
        };
        assert_eq!(
            RunStatus::CheckingDownload(half).to_string(),
            "Checking the download 50%"
        );
        assert_eq!(
            RunStatus::CheckingDownload(VerifyProgress::default()).to_string(),
            "Checking the download"
        );
    }
}
