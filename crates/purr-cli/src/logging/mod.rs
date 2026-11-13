//! The diagnostic log: `tracing` events, shown on stderr with `-v` and appended to `--log-file`
//! at a level of its own.

#[cfg(test)]
mod capture;
pub mod file;
pub mod format;
pub mod level;

use crate::APP_NAME;
use crate::cli::Cli;
use crate::color::Stream;
use crate::global;
use crate::output::{self, channel::Channel};
use crate::verbosity::Verbosity;
use file::{LogFile, LogFileError};
use format::LabelFormat;
use level::LogLevel;
use std::fmt::Display;
use std::io;
use tracing::{Level, Metadata, Subscriber};
use tracing_subscriber::filter::{FilterExt as _, LevelFilter, filter_fn};
use tracing_subscriber::fmt::{MakeWriter, time::SystemTime, time::Uptime};
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::registry::Registry;
use tracing_subscriber::{EnvFilter, Layer as _};

/// Target of the event that records the error ending the run; the terminal shows that error on
/// its own channel, so only the log file gets this one.
pub const FATAL_TARGET: &str = "purr::fatal";

type BoxedLayer = Box<dyn tracing_subscriber::Layer<Registry> + Send + Sync>;

#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum LoggingError {
    #[error(transparent)]
    #[diagnostic(transparent)]
    LogFile(#[from] LogFileError),
    #[error("RUST_LOG is not a valid filter: {0}")]
    Filter(#[from] tracing_subscriber::filter::FromEnvError),
    #[error("a logger is already installed: {0}")]
    Install(#[from] tracing::subscriber::SetGlobalDefaultError),
}

/// What the terminal shows of the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalLogs {
    Off,
    /// Info and above from this program and `purr-core`, whisper and ggml included
    Verbose,
    /// Everything, narrowed by `RUST_LOG` when it is set
    Everything,
}

impl TerminalLogs {
    pub const fn from_verbosity(verbosity: Verbosity) -> Self {
        if !verbosity.shows_diagnostics() {
            Self::Off
        } else if verbosity.is_debug() {
            Self::Everything
        } else {
            Self::Verbose
        }
    }
}

pub struct LogConfig {
    pub terminal: TerminalLogs,
    pub terminal_ansi: bool,
    pub file: Option<(LogFile, LogLevel)>,
}

fn shown_when_verbose(metadata: &Metadata<'_>) -> bool {
    let target = metadata.target();
    *metadata.level() <= Level::INFO
        && target != FATAL_TARGET
        && [APP_NAME, purr_core::PKG_NAME]
            .iter()
            .any(|prefix| target.starts_with(prefix))
}

fn terminal_layer<W>(
    logs: TerminalLogs,
    ansi: bool,
    writer: W,
) -> Result<Option<BoxedLayer>, LoggingError>
where
    W: for<'a> MakeWriter<'a> + Send + Sync + 'static,
{
    match logs {
        TerminalLogs::Off => Ok(None),
        TerminalLogs::Verbose => Ok(Some(
            tracing_subscriber::fmt::layer()
                .with_ansi(ansi)
                .event_format(LabelFormat)
                .with_writer(writer)
                .with_filter(filter_fn(shown_when_verbose))
                .boxed(),
        )),
        TerminalLogs::Everything => {
            let env = EnvFilter::builder()
                .with_default_directive(LevelFilter::TRACE.into())
                .from_env()?;
            Ok(Some(
                tracing_subscriber::fmt::layer()
                    .with_ansi(ansi)
                    .with_timer(Uptime::default())
                    .with_writer(writer)
                    .with_filter(env.and(filter_fn(|metadata| metadata.target() != FATAL_TARGET)))
                    .boxed(),
            ))
        }
    }
}

fn file_layer(file: LogFile, level: LogLevel) -> BoxedLayer {
    tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_timer(SystemTime)
        .with_writer(file)
        .with_filter(level.filter())
        .boxed()
}

/// The subscriber for `config`; `terminal_writer` is where the terminal's lines go.
pub fn subscriber<W>(
    config: LogConfig,
    terminal_writer: W,
) -> Result<impl Subscriber + Send + Sync, LoggingError>
where
    W: for<'a> MakeWriter<'a> + Send + Sync + 'static,
{
    let mut layers: Vec<BoxedLayer> = Vec::new();
    layers.extend(terminal_layer(
        config.terminal,
        config.terminal_ansi,
        terminal_writer,
    )?);
    layers.extend(config.file.map(|(file, level)| file_layer(file, level)));
    Ok(Registry::default().with(layers))
}

/// Terminal log lines go through the output layer, so they are routed like everything else.
pub struct TerminalLog;

pub struct TerminalLogWriter;

impl<'a> MakeWriter<'a> for TerminalLog {
    type Writer = TerminalLogWriter;

    fn make_writer(&'a self) -> Self::Writer {
        TerminalLogWriter
    }
}

impl io::Write for TerminalLogWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        output::current().write_text(Channel::Log, &String::from_utf8_lossy(bytes))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Opens the log file and installs the subscriber; a log file that cannot be opened is an error
/// before anything else happens.
pub fn init(cli: &Cli) -> Result<(), LoggingError> {
    let file = cli.log_file.as_deref().map(LogFile::open).transpose()?;
    let config = LogConfig {
        terminal: TerminalLogs::from_verbosity(cli.verbosity),
        terminal_ansi: global::colors().enabled(Stream::Stderr),
        file: file.map(|file| (file, cli.log_level.unwrap_or_default())),
    };
    tracing::subscriber::set_global_default(subscriber(config, TerminalLog)?)?;
    Ok(())
}

/// Puts the error that ends the run in the log file.
pub fn record_fatal(error: &dyn Display) {
    tracing::error!(target: FATAL_TARGET, "{error}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verbosity::VerbosityLevel;
    use capture::Capture;

    fn run(config: LogConfig, log: impl FnOnce()) -> Result<Capture, LoggingError> {
        let terminal = Capture::default();
        let subscriber = subscriber(config, terminal.clone())?;
        tracing::subscriber::with_default(subscriber, log);
        Ok(terminal)
    }

    fn open_log(dir: &std::path::Path) -> Result<(LogFile, std::path::PathBuf), LoggingError> {
        let path = dir.join("purr.log");
        Ok((LogFile::open(&path)?, path))
    }

    fn without_timestamps(text: &str) -> Vec<String> {
        text.lines()
            .map(|line| {
                line.split_whitespace()
                    .skip(1)
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect()
    }

    fn emit_every_level() {
        tracing::error!(target: "purr", "e");
        tracing::warn!(target: "purr", "w");
        tracing::info!(target: "purr", "i");
        tracing::debug!(target: "purr", "d");
        tracing::trace!(target: "purr", "t");
    }

    #[test]
    fn the_log_file_has_its_own_level_whatever_the_terminal_shows()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let (file, path) = open_log(dir.path())?;
        let terminal = run(
            LogConfig {
                terminal: TerminalLogs::Off,
                terminal_ansi: true,
                file: Some((file, LogLevel::Debug)),
            },
            emit_every_level,
        )?;
        assert_eq!(terminal.text(), "");
        assert_eq!(
            without_timestamps(&std::fs::read_to_string(&path)?),
            [
                "ERROR purr: e",
                "WARN purr: w",
                "INFO purr: i",
                "DEBUG purr: d"
            ]
        );
        Ok(())
    }

    #[test]
    fn the_log_file_is_plain_text_even_when_the_terminal_has_colours()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let (file, path) = open_log(dir.path())?;
        let terminal = run(
            LogConfig {
                terminal: TerminalLogs::Verbose,
                terminal_ansi: true,
                file: Some((file, LogLevel::Info)),
            },
            || tracing::info!(target: "purr", "hello"),
        )?;
        assert_eq!(terminal.text(), "\u{1b}[34;1minfo\u{1b}[0m: hello\n");
        let written = std::fs::read(&path)?;
        assert_eq!(written.iter().position(|byte| *byte == 0x1b), None);
        Ok(())
    }

    #[test]
    fn the_terminal_level_does_not_limit_the_log_file_and_the_other_way_round()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let (file, path) = open_log(dir.path())?;
        let terminal = run(
            LogConfig {
                terminal: TerminalLogs::Verbose,
                terminal_ansi: false,
                file: Some((file, LogLevel::Warn)),
            },
            emit_every_level,
        )?;
        assert_eq!(terminal.text(), "error: e\nwarn: w\ninfo: i\n");
        assert_eq!(
            without_timestamps(&std::fs::read_to_string(&path)?),
            ["ERROR purr: e", "WARN purr: w"]
        );
        Ok(())
    }

    #[test]
    fn verbose_shows_this_program_and_purr_core_but_not_the_fatal_record()
    -> Result<(), LoggingError> {
        let terminal = run(
            LogConfig {
                terminal: TerminalLogs::Verbose,
                terminal_ansi: false,
                file: None,
            },
            || {
                tracing::info!(target: "purr", "app");
                tracing::info!(target: "purr-core::logging::trampoline::ggml", "ggml");
                tracing::info!(target: "hyper", "dependency");
                tracing::error!(target: FATAL_TARGET, "fatal");
            },
        )?;
        assert_eq!(terminal.text(), "info: app\ninfo: ggml\n");
        Ok(())
    }

    #[test]
    fn the_fatal_record_reaches_the_log_file() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let (file, path) = open_log(dir.path())?;
        run(
            LogConfig {
                terminal: TerminalLogs::Verbose,
                terminal_ansi: false,
                file: Some((file, LogLevel::Info)),
            },
            || record_fatal(&"boom"),
        )?;
        assert_eq!(
            without_timestamps(&std::fs::read_to_string(&path)?),
            ["ERROR purr::fatal: boom"]
        );
        Ok(())
    }

    #[test]
    fn the_terminal_shows_the_log_from_the_first_verbose_flag() {
        let table = [
            (VerbosityLevel::NORMAL, TerminalLogs::Off),
            (VerbosityLevel::VERBOSE, TerminalLogs::Verbose),
            (VerbosityLevel::DEBUG, TerminalLogs::Everything),
            (VerbosityLevel::TRACE, TerminalLogs::Everything),
        ];
        for (verbose, expected) in table {
            assert_eq!(
                TerminalLogs::from_verbosity(Verbosity { verbose }),
                expected
            );
        }
    }
}
