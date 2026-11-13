use purr_core::whisper::logging::{GGML_LOG_TARGET, WHISPER_LOG_TARGET};
use std::{fmt, ops::Deref};
use tracing::Level;
use tracing_core::{Event, Subscriber};
use tracing_subscriber::fmt::{
    FmtContext, FormattedFields,
    format::{self, FormatEvent, FormatFields},
};
use tracing_subscriber::registry::LookupSpan;

use crate::APP_NAME;
use crate::theme::Role;

pub struct MyFormatter {
    _verbosity: Verbosity,
    filter: Box<dyn Fn(&Event<'_>) -> bool + Send + Sync>,
}

impl MyFormatter {
    pub fn new(verbosity: Verbosity) -> Self {
        let filter: Box<dyn Fn(&Event<'_>) -> bool + Send + Sync> = match *verbosity.verbose {
            VerbosityLevel::NORMAL_VALUE => Box::new(|event: &Event<'_>| {
                *event.metadata().level() <= Level::INFO
                    && ![WHISPER_LOG_TARGET, GGML_LOG_TARGET].contains(&event.metadata().target())
                    && [APP_NAME, purr_core::PKG_NAME]
                        .iter()
                        .any(|t| event.metadata().target().starts_with(t))
            }),
            VerbosityLevel::VERBOSE_VALUE => Box::new(|event: &Event<'_>| {
                [
                    APP_NAME,
                    purr_core::PKG_NAME,
                    WHISPER_LOG_TARGET,
                    GGML_LOG_TARGET,
                ]
                .iter()
                .any(|t| event.metadata().target().starts_with(t))
            }),
            VerbosityLevel::DEBUG_VALUE => {
                Box::new(|event: &Event<'_>| *event.metadata().level() <= Level::DEBUG)
            }
            VerbosityLevel::TRACE_VALUE => {
                Box::new(|event: &Event<'_>| *event.metadata().level() <= Level::TRACE)
            }
            _ => Box::new(|_: &Event<'_>| true),
        };
        Self {
            _verbosity: verbosity,
            filter,
        }
    }
}

impl<S, N> FormatEvent<S, N> for MyFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        // Format values from the event's's metadata:
        let metadata = event.metadata();
        if !(self.filter)(event) {
            return Ok(());
        }
        let (role, label) = match *metadata.level() {
            Level::TRACE => (Role::Trace, "trace"),
            Level::DEBUG => (Role::Debug, "debug"),
            Level::INFO => (Role::Info, "info"),
            Level::WARN => (Role::Warning, "warn"),
            Level::ERROR => (Role::Error, "error"),
        };
        let ansi = writer.has_ansi_escapes();
        write!(&mut writer, "{}: ", role.paint(label).fixed(ansi))?;

        // Format all the spans in the event's span context.
        if let Some(scope) = ctx.event_scope() {
            for span in scope.from_root() {
                write!(writer, "{}", span.name())?;

                // `FormattedFields` is a formatted representation of the span's
                // fields, which is stored in its extensions by the `fmt` layer's
                // `new_span` method. The fields will have been formatted
                // by the same field formatter that's provided to the event
                // formatter in the `FmtContext`.
                let ext = span.extensions();
                let fields = &ext
                    .get::<FormattedFields<N>>()
                    .expect("will never be `None`");

                // Skip formatting the fields if the span had no fields.
                if !fields.is_empty() {
                    write!(writer, "{{{fields}}}")?;
                }
                write!(writer, ": ")?;
            }
        }

        // Write fields on the event
        ctx.field_format().format_fields(writer.by_ref(), event)?;

        writeln!(writer)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, clap::Args)]
pub struct Verbosity {
    #[clap(flatten)]
    pub verbose: VerbosityLevel,
}

impl Deref for Verbosity {
    type Target = VerbosityLevel;

    fn deref(&self) -> &Self::Target {
        &self.verbose
    }
}

#[allow(dead_code)]
impl Verbosity {
    pub const fn is_verbose(&self) -> bool {
        self.verbose.verbose >= VerbosityLevel::VERBOSE.verbose
    }

    pub const fn is_debug(&self) -> bool {
        self.verbose.verbose >= VerbosityLevel::DEBUG.verbose
    }

    pub const fn is_trace(&self) -> bool {
        self.verbose.verbose >= VerbosityLevel::TRACE.verbose
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, clap::Args)]

pub struct VerbosityLevel {
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    /// Verbose output (-v, -vv, -vvv for more verbosity)
    verbose: u8,
}

#[allow(dead_code)]
impl VerbosityLevel {
    pub const TRACE_VALUE: u8 = 3;
    pub const DEBUG_VALUE: u8 = 2;
    pub const VERBOSE_VALUE: u8 = 1;
    pub const NORMAL_VALUE: u8 = 0;
    pub const TRACE: Self = Self {
        verbose: Self::TRACE_VALUE,
    };
    pub const DEBUG: Self = Self {
        verbose: Self::DEBUG_VALUE,
    };
    pub const VERBOSE: Self = Self {
        verbose: Self::VERBOSE_VALUE,
    };
    pub const NORMAL: Self = Self {
        verbose: Self::NORMAL_VALUE,
    };
}

impl Deref for VerbosityLevel {
    type Target = u8;

    fn deref(&self) -> &Self::Target {
        &self.verbose
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::fmt::MakeWriter;

    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("lock").extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for Capture {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    fn logged(ansi: bool, log: impl FnOnce()) -> String {
        let capture = Capture::default();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(ansi)
            .event_format(MyFormatter::new(Verbosity {
                verbose: VerbosityLevel::NORMAL,
            }))
            .with_writer(capture.clone())
            .finish();
        tracing::subscriber::with_default(subscriber, log);
        let bytes = capture.0.lock().expect("lock").clone();
        String::from_utf8(bytes).expect("utf-8")
    }

    #[test]
    fn a_log_line_without_colours_is_plain_text() {
        let line = logged(false, || {
            tracing::info!(target: "purr", "Input: {} (file)", "jfk.mp3");
        });
        assert_eq!(line, "info: Input: jfk.mp3 (file)\n");
    }

    #[test]
    fn a_log_line_with_colours_styles_only_its_level() {
        let line = logged(true, || {
            tracing::info!(target: "purr", "Input: {} (file)", "jfk.mp3");
        });
        assert_eq!(line, "\u{1b}[34;1minfo\u{1b}[0m: Input: jfk.mp3 (file)\n");
    }

    #[test]
    fn every_level_has_its_role() {
        let line = logged(true, || {
            tracing::warn!(target: "purr", "careful");
            tracing::error!(target: "purr", "broken");
        });
        assert_eq!(
            line,
            "\u{1b}[33;1mwarn\u{1b}[0m: careful\n\u{1b}[31;1merror\u{1b}[0m: broken\n"
        );
    }
}
