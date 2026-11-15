use crate::language_arg::LanguageProblem;
use crate::logging::LoggingError;
use crate::output::OutputError;
use indicatif::style::TemplateError;
use purr_core::accel::selector::SelectorError;
use purr_core::error::Error as CoreError;
use std::io;
use std::path::PathBuf;

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
        code = "purr::models::verification_failed",
        help = "Download the affected models again with 'models download <model> --force'"
    )]
    VerificationFailed { failed: usize },
    #[oopsie(
        "Nothing was transcribed",
        code = "purr::transcribe::nothing_transcribed",
        help = "Check that it is a valid audio file"
    )]
    NothingTranscribed,
    #[oopsie(
        "No audio file or URL given",
        code = "purr::input::missing",
        help = "Pass an audio file path or a URL"
    )]
    NoAudioInput,
    #[oopsie("Not a file path or URL: '{input}'", code = "purr::input::invalid")]
    InvalidAudioInput { input: String },
    #[oopsie(
        "Audio file not found: {}",
        path.display(),
        code = "purr::input::file_not_found"
    )]
    AudioFileNotFound { path: PathBuf },
    #[oopsie(
        "Model file not found at: {}",
        path.display(),
        code = "purr::models::file_not_found"
    )]
    ModelFileNotFound { path: PathBuf },
    #[oopsie(
        "Model {model} needs the {engine}, which this build does not include",
        code = "purr::models::engine_not_built"
    )]
    EngineNotBuilt { model: String, engine: String },
    #[oopsie(
        "Model {name} is not downloaded",
        code = "purr::models::not_downloaded",
        help = "Download it with: {program} models download {id}"
    )]
    ModelNotDownloaded {
        name: String,
        id: String,
        program: &'static str,
    },
    #[oopsie(
        "No model downloaded",
        code = "purr::models::none_downloaded",
        help = "Download one with: {program} models download {id}"
    )]
    NoModelDownloaded { id: String, program: &'static str },
    #[oopsie(
        "No model specified and no downloaded models found",
        code = "purr::models::none_available"
    )]
    NoModelAvailable,
    #[oopsie(
        "No per-user data directory is known",
        code = "purr::runtime::no_data_dir",
        help = "Set PURR_RUNTIME_DIR"
    )]
    NoDataDir,
    #[oopsie(
        "Failed to read the answer from stdin",
        code = "purr::input::stdin_failed"
    )]
    ReadAnswer { source: io::Error },
    #[oopsie(
        "Failed to serialize the result as JSON",
        code = "purr::output::json_failed"
    )]
    SerializeJson { source: serde_json::Error },
    #[oopsie(
        "Invalid progress display template",
        code = "purr::output::progress_template"
    )]
    ProgressTemplate { source: TemplateError },
    #[oopsie(
        "Failed to write {}",
        path.display(),
        code = "purr::output::write_failed"
    )]
    WriteOutput { path: PathBuf, source: io::Error },
    #[oopsie(
        "Failed to read the current directory",
        code = "purr::env::current_dir_failed"
    )]
    CurrentDir { source: io::Error },
}

pub type CliResult<T> = Result<T, CliError>;
