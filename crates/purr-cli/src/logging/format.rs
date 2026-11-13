//! The terminal's log line: the level as a coloured label, then the spans and the message.

use crate::theme::Role;
use std::fmt;
use tracing::Level;
use tracing_core::{Event, Subscriber};
use tracing_subscriber::fmt::{
    FmtContext, FormattedFields,
    format::{self, FormatEvent, FormatFields},
};
use tracing_subscriber::registry::LookupSpan;

pub struct LabelFormat;

impl<S, N> FormatEvent<S, N> for LabelFormat
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
        let (role, label) = match *event.metadata().level() {
            Level::TRACE => (Role::Trace, "trace"),
            Level::DEBUG => (Role::Debug, "debug"),
            Level::INFO => (Role::Info, "info"),
            Level::WARN => (Role::Warning, "warn"),
            Level::ERROR => (Role::Error, "error"),
        };
        let ansi = writer.has_ansi_escapes();
        write!(&mut writer, "{}: ", role.paint(label).fixed(ansi))?;

        if let Some(scope) = ctx.event_scope() {
            for span in scope.from_root() {
                write!(writer, "{}", span.name())?;
                let ext = span.extensions();
                if let Some(fields) = ext.get::<FormattedFields<N>>()
                    && !fields.is_empty()
                {
                    write!(writer, "{{{fields}}}")?;
                }
                write!(writer, ": ")?;
            }
        }

        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::capture::Capture;

    fn logged(ansi: bool, log: impl FnOnce()) -> String {
        let capture = Capture::default();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(ansi)
            .event_format(LabelFormat)
            .with_writer(capture.clone())
            .finish();
        tracing::subscriber::with_default(subscriber, log);
        capture.text()
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
