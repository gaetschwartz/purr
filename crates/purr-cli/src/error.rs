use crate::language_arg::LanguageProblem;
use crate::logging::LoggingError;
use crate::output::OutputError;
use oopsie::Welp;
use purr_core::accel::selector::SelectorError;
use purr_core::error::Error as CoreError;

#[oopsie::oopsie]
pub enum CliError {
    #[oopsie(transparent)]
    Whisper { source: CoreError },
    #[oopsie(transparent)]
    Selector { source: SelectorError },
    #[oopsie(transparent)]
    Language { source: LanguageProblem },
    #[oopsie(transparent)]
    Output { source: OutputError },
    #[oopsie(transparent)]
    Logging { source: LoggingError },
    #[oopsie(
        "{failed} model(s) failed verification",
        help = "Download the affected models again with 'models download <model> --force'"
    )]
    VerificationFailed { failed: usize },
    #[oopsie(
        "Nothing was transcribed",
        help = "Check that it is a valid audio file"
    )]
    NothingTranscribed,
    #[oopsie(transparent)]
    Other { source: Welp },
}

impl CliError {
    #[track_caller]
    pub fn message(message: impl Into<String>) -> Self {
        Self::Other {
            source: Welp::new(message),
        }
    }
}

pub type CliResult<T> = Result<T, CliError>;
