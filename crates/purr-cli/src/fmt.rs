use owo_colors::OwoColorize;
use purr_core::whisper::logging::{GGML_LOG_TARGET, WHISPER_LOG_TARGET};
use std::{fmt, ops::Deref};
use tracing::Level;
use tracing_core::{Event, Subscriber};
use tracing_subscriber::fmt::{
    format::{self, FormatEvent, FormatFields},
    FmtContext, FormattedFields,
};
use tracing_subscriber::registry::LookupSpan;

use crate::APP_NAME;

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
                if *event.metadata().level() <= Level::INFO {
                    return false;
                }
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
        match *metadata.level() {
            Level::TRACE => write!(
                &mut writer,
                "{}: ",
                "trace".fg_rgb::<255, 105, 180>().bold()
            ),
            Level::DEBUG => write!(&mut writer, "{}: ", "debug".green().bold()),
            Level::INFO => write!(&mut writer, "{}: ", "info".blue().bold()),
            Level::WARN => write!(&mut writer, "{}: ", "warn".yellow().bold()),
            Level::ERROR => write!(&mut writer, "{}: ", "error".red().bold()),
        }?;

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
    pub const TRACE: VerbosityLevel = VerbosityLevel {
        verbose: Self::TRACE_VALUE,
    };
    pub const DEBUG: VerbosityLevel = VerbosityLevel {
        verbose: Self::DEBUG_VALUE,
    };
    pub const VERBOSE: VerbosityLevel = VerbosityLevel {
        verbose: Self::VERBOSE_VALUE,
    };
    pub const NORMAL: VerbosityLevel = VerbosityLevel {
        verbose: Self::NORMAL_VALUE,
    };
}

impl Deref for VerbosityLevel {
    type Target = u8;

    fn deref(&self) -> &Self::Target {
        &self.verbose
    }
}
