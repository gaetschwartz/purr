use crate::color::Stream;
use crate::error::CliError;
use crate::global;
use oopsie::{Report, Welp, WelpResultExt as _};
use purr_core::error::{Error as CoreError, ManifestError};

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
    Chain,
    Coded,
    Help,
}

impl ThrowKind {
    pub fn raise(self) -> CliError {
        match self {
            Self::Message => CliError::message("a plain message error"),
            Self::Io => std::fs::read("/definitely/not/here")
                .welp_context("could not read the sample file")
                .unwrap_err()
                .into(),
            Self::Chain => Welp::wrap(
                Welp::wrap(
                    std::io::Error::other("the disk went away"),
                    "could not save the transcript",
                ),
                "the run failed",
            )
            .into(),
            Self::Coded => CoreError::from(ManifestError::NoPreferredModel).into(),
            Self::Help => CliError::VerificationFailed { failed: 3 },
        }
    }
}
