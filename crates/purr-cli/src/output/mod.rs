//! Everything `purr-cli` prints goes through here, on a named channel.
//!
//! The channel decides the stream (see `channel::route`) and the stream decides the colours: text
//! for stderr is formatted inside `scope::on_stderr`, so call sites never think about it. Progress
//! bars live in one `MultiProgress` and every write suspends them, so a line never lands in the
//! middle of a bar. `println!`, `eprintln!` and `dbg!` are denied in this crate and
//! `std::io::stdout`/`stderr` are only reachable from `sink`.

pub mod channel;
pub mod mode;
pub mod sink;

use crate::color::Stream;
use crate::global;
use crate::scope;
use crate::verbosity::Verbosity;
use channel::{Channel, route};
use indicatif::{MultiProgress, ProgressBar};
use mode::Mode;
use sink::{ProcessSink, Sink};
use std::fmt;
use std::io;
use std::sync::OnceLock;

#[derive(Debug, thiserror::Error, miette::Diagnostic)]
#[error("cannot write to {}: {source}", stream.name())]
pub struct OutputError {
    pub stream: Stream,
    #[source]
    pub source: io::Error,
}

pub struct Output {
    mode: Mode,
    verbosity: Verbosity,
    sink: Box<dyn Sink>,
    progress: MultiProgress,
}

impl Output {
    pub fn new(mode: Mode, verbosity: Verbosity, sink: Box<dyn Sink>) -> Self {
        Self {
            mode,
            verbosity,
            sink,
            progress: MultiProgress::with_draw_target(mode.draw_target()),
        }
    }

    pub const fn mode(&self) -> Mode {
        self.mode
    }

    /// Writes `args` on `channel`, or drops them when the channel has no stream in this mode.
    pub fn write(&self, channel: Channel, args: fmt::Arguments<'_>) -> Result<(), OutputError> {
        let Some(stream) = route(channel, self.mode, self.verbosity) else {
            return Ok(());
        };
        let text = match stream {
            Stream::Stdout => args.to_string(),
            Stream::Stderr => scope::on_stderr(|| args.to_string()),
        };
        self.deliver(stream, &text)
            .map_err(|source| OutputError { stream, source })
    }

    /// Writes text that is already formatted, as the diagnostic log does.
    pub fn write_text(&self, channel: Channel, text: &str) -> io::Result<()> {
        route(channel, self.mode, self.verbosity)
            .map_or(Ok(()), |stream| self.deliver(stream, text))
    }

    fn deliver(&self, stream: Stream, text: &str) -> io::Result<()> {
        if self.progress.is_hidden() {
            self.sink.write(stream, text)
        } else {
            self.progress.suspend(|| self.sink.write(stream, text))
        }
    }

    /// Puts a bar under the output's care: drawn on stdout in interactive mode, invisible in plain
    /// mode.
    pub fn add_bar(&self, bar: ProgressBar) -> ProgressBar {
        self.progress.add(bar)
    }
}

static OUTPUT: OnceLock<Output> = OnceLock::new();

/// The process-wide output, built from the global arguments on first use.
pub fn current() -> &'static Output {
    OUTPUT.get_or_init(|| Output::new(global::mode(), global::verbosity(), Box::new(ProcessSink)))
}

pub fn is_interactive() -> bool {
    !current().mode().is_plain()
}

pub fn add_bar(bar: ProgressBar) -> ProgressBar {
    current().add_bar(bar)
}

/// The command's product. Failing to write it fails the command.
pub fn result(args: fmt::Arguments<'_>) -> Result<(), OutputError> {
    current().write(Channel::Result, args)
}

/// A message for a person. A failed write has nowhere to be reported and is dropped.
pub fn status(args: fmt::Arguments<'_>) {
    best_effort(current().write(Channel::Status, args));
}

/// The error that ends the run.
pub fn error(args: fmt::Arguments<'_>) {
    best_effort(current().write(Channel::Error, args));
}

/// A diagnostic line the user asked for with a flag.
pub fn trace(args: fmt::Arguments<'_>) {
    best_effort(current().write(Channel::Trace, args));
}

/// Dropping the failure is the point: the line was a courtesy and stderr may be what is broken.
fn best_effort(written: Result<(), OutputError>) {
    if written.is_err() {
        tracing::debug!("an output line could not be written");
    }
}

#[macro_export]
macro_rules! resultln {
    () => {
        $crate::output::result(::std::format_args!("\n"))
    };
    ($($arg:tt)*) => {
        $crate::output::result(::std::format_args!("{}\n", ::std::format_args!($($arg)*)))
    };
}

#[macro_export]
macro_rules! result {
    ($($arg:tt)*) => {
        $crate::output::result(::std::format_args!($($arg)*))
    };
}

#[macro_export]
macro_rules! statusln {
    () => {
        $crate::output::status(::std::format_args!("\n"))
    };
    ($($arg:tt)*) => {
        $crate::output::status(::std::format_args!("{}\n", ::std::format_args!($($arg)*)))
    };
}

#[macro_export]
macro_rules! status {
    ($($arg:tt)*) => {
        $crate::output::status(::std::format_args!($($arg)*))
    };
}

#[macro_export]
macro_rules! errorln {
    ($($arg:tt)*) => {
        $crate::output::error(::std::format_args!("{}\n", ::std::format_args!($($arg)*)))
    };
}

#[macro_export]
macro_rules! traceln {
    ($($arg:tt)*) => {
        $crate::output::trace(::std::format_args!("{}\n", ::std::format_args!($($arg)*)))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verbosity::VerbosityLevel;
    use sink::memory::MemorySink;

    fn output(
        plain_flag: bool,
        stdout_terminal: bool,
        verbose: VerbosityLevel,
    ) -> (Output, MemorySink) {
        let sink = MemorySink::default();
        let mode = Mode::decide(plain_flag, stdout_terminal);
        let output = Output::new(mode, Verbosity { verbose }, Box::new(sink.clone()));
        (output, sink)
    }

    fn write_everywhere(output: &Output) -> Result<(), OutputError> {
        output.write(Channel::Result, format_args!("result;"))?;
        output.write(Channel::Status, format_args!("status;"))?;
        output.write(Channel::Error, format_args!("error;"))?;
        output.write(Channel::Trace, format_args!("trace;"))?;
        output.write(Channel::Log, format_args!("log;"))?;
        Ok(())
    }

    #[test]
    fn a_terminal_gets_the_result_and_the_status_on_stdout_and_the_error_on_stderr()
    -> Result<(), OutputError> {
        let (output, sink) = output(false, true, VerbosityLevel::NORMAL);
        write_everywhere(&output)?;
        assert_eq!(sink.stdout(), "result;status;");
        assert_eq!(sink.stderr(), "error;trace;");
        Ok(())
    }

    #[test]
    fn a_pipe_gets_only_the_result_on_stdout() -> Result<(), OutputError> {
        let (output, sink) = output(false, false, VerbosityLevel::NORMAL);
        write_everywhere(&output)?;
        assert_eq!(sink.stdout(), "result;");
        assert_eq!(sink.stderr(), "error;trace;");
        Ok(())
    }

    #[test]
    fn the_plain_flag_on_a_terminal_acts_like_a_pipe() -> Result<(), OutputError> {
        let (output, sink) = output(true, true, VerbosityLevel::NORMAL);
        write_everywhere(&output)?;
        assert_eq!(sink.stdout(), "result;");
        assert_eq!(sink.stderr(), "error;trace;");
        Ok(())
    }

    #[test]
    fn verbose_adds_the_diagnostic_log_to_stderr_in_both_modes() -> Result<(), OutputError> {
        for (plain_flag, terminal, stdout) in
            [(false, true, "result;status;"), (false, false, "result;")]
        {
            let (output, sink) = output(plain_flag, terminal, VerbosityLevel::VERBOSE);
            write_everywhere(&output)?;
            assert_eq!(
                sink.stdout(),
                stdout,
                "--plain={plain_flag} terminal={terminal}"
            );
            assert_eq!(sink.stderr(), "error;trace;log;");
        }
        Ok(())
    }

    #[test]
    fn already_formatted_log_text_follows_the_log_route() -> io::Result<()> {
        let (quiet, quiet_sink) = output(false, true, VerbosityLevel::NORMAL);
        quiet.write_text(Channel::Log, "info: x\n")?;
        let (verbose, verbose_sink) = output(false, true, VerbosityLevel::VERBOSE);
        verbose.write_text(Channel::Log, "info: x\n")?;
        assert_eq!(
            (quiet_sink.stderr(), verbose_sink.stderr()),
            (String::new(), "info: x\n".to_owned())
        );
        Ok(())
    }

    struct CurrentStream;

    impl fmt::Display for CurrentStream {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{:?}", scope::current())
        }
    }

    #[test]
    fn text_is_formatted_for_the_stream_it_goes_to() -> Result<(), OutputError> {
        let (output, sink) = output(false, true, VerbosityLevel::NORMAL);
        output.write(Channel::Error, format_args!("{CurrentStream}"))?;
        output.write(Channel::Result, format_args!("{CurrentStream}"))?;
        assert_eq!(
            (sink.stderr(), sink.stdout()),
            ("Stderr".to_owned(), "Stdout".to_owned())
        );
        Ok(())
    }
}
