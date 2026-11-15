use crate::color::Stream;
use crate::error::CliError;
use crate::global;
use oopsie::Report;
use purr_core::error::{Error as CoreError, ManifestError};
use std::path::PathBuf;

pub fn render(error: CliError) -> String {
    let report = Report::new(error);
    let report = if global::colors().enabled(Stream::Stderr) {
        report.force_colors()
    } else {
        report.no_colors()
    };
    report.to_string().trim_end().to_owned()
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum ThrowKind {
    Message,
    Io,
    Coded,
    Help,
}

impl ThrowKind {
    pub fn raise(self) -> CliError {
        match self {
            Self::Message => CliError::NoAudioInput,
            Self::Io => CliError::WriteOutput {
                path: PathBuf::from("transcript.txt"),
                source: std::io::Error::other("the disk went away"),
            },
            Self::Coded => CoreError::from(ManifestError::NoPreferredModel).into(),
            Self::Help => CliError::VerificationFailed { failed: 3 },
        }
    }
}
