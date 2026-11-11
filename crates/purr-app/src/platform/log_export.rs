//! Turns captured log entries into the text, JSON and CSV files a user can save.

use super::logging::LogEntry;
use miette::IntoDiagnostic as _;
use std::sync::Arc;

/// A file format for exported logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogExportFormat {
    Text,
    Json,
    Csv,
}

impl LogExportFormat {
    /// Every format, in the order a menu lists them.
    pub const ALL: [Self; 3] = [Self::Text, Self::Json, Self::Csv];

    /// The name shown in the export menu.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Text => "Text",
            Self::Json => "JSON",
            Self::Csv => "CSV",
        }
    }

    /// The file extension, dot included.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Text => ".log",
            Self::Json => ".json",
            Self::Csv => ".csv",
        }
    }

    /// The file name offered when saving.
    #[must_use]
    pub fn file_name(self) -> String {
        format!("purr-logs{}", self.extension())
    }

    /// The media type of the file.
    #[must_use]
    pub const fn content_type(self) -> &'static str {
        match self {
            Self::Text => "text/plain;charset=utf-8",
            Self::Json => "application/json",
            Self::Csv => "text/csv;charset=utf-8",
        }
    }
}

/// Quotes a CSV field when it holds a separator, a quote or a line break.
fn csv_field(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_owned()
    }
}

fn text_line(entry: &LogEntry, time: &str) -> String {
    format!(
        "{time} [{:<5}] {}: {}",
        entry.level.to_string(),
        entry.short_target(),
        entry.message
    )
}

fn csv_row(entry: &LogEntry, time: &str) -> String {
    [
        csv_field(time),
        csv_field(&entry.level.to_string()),
        csv_field(&entry.target),
        csv_field(&entry.message),
        csv_field(entry.file.as_deref().unwrap_or_default()),
        entry.line.map(|line| line.to_string()).unwrap_or_default(),
    ]
    .join(",")
}

/// Renders `entries` in `format`; `time` writes each entry's clock time for the text and CSV files.
pub fn format_entries(
    format: LogExportFormat,
    entries: &[Arc<LogEntry>],
    time: impl Fn(&LogEntry) -> String,
) -> miette::Result<String> {
    match format {
        LogExportFormat::Text => Ok(entries
            .iter()
            .map(|entry| text_line(entry, &time(entry)))
            .collect::<Vec<_>>()
            .join("\n")),
        LogExportFormat::Json => {
            let plain: Vec<&LogEntry> = entries.iter().map(Arc::as_ref).collect();
            serde_json::to_string_pretty(&plain).into_diagnostic()
        }
        LogExportFormat::Csv => {
            let rows = entries
                .iter()
                .map(|entry| csv_row(entry, &time(entry)))
                .collect::<Vec<_>>();
            let mut csv = String::from("Timestamp,Level,Target,Message,File,Line\n");
            for row in rows {
                csv.push_str(&row);
                csv.push('\n');
            }
            Ok(csv)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::logging::SerdeTracingLevel;

    fn entry(
        id: u64,
        level: SerdeTracingLevel,
        target: &str,
        message: &str,
        file: Option<&str>,
    ) -> Arc<LogEntry> {
        Arc::new(LogEntry {
            id,
            timestamp: id * 1000,
            level,
            target: target.to_owned(),
            message: message.to_owned(),
            fields: vec![("key".to_owned(), "value".to_owned())],
            file: file.map(str::to_owned),
            line: file.map(|_| 7),
        })
    }

    fn sample() -> Vec<Arc<LogEntry>> {
        vec![
            entry(
                1,
                SerdeTracingLevel::INFO,
                "purr_app::views::logs",
                "plain",
                Some("logs.rs"),
            ),
            entry(
                2,
                SerdeTracingLevel::WARN,
                "whisper",
                "has, comma and \"quotes\"\nand a line break",
                None,
            ),
        ]
    }

    fn clock(entry: &LogEntry) -> String {
        format!("t{}", entry.id)
    }

    #[test]
    fn text_lists_one_line_per_entry() -> miette::Result<()> {
        let text = format_entries(LogExportFormat::Text, &sample(), clock)?;
        let expected = [
            "t1 [INFO ] logs: plain",
            "t2 [WARN ] whisper: has, comma and \"quotes\"\nand a line break",
        ]
        .join("\n");
        assert_eq!(text, expected);
        Ok(())
    }

    #[test]
    fn csv_quotes_every_field_that_needs_it() -> miette::Result<()> {
        let csv = format_entries(LogExportFormat::Csv, &sample(), clock)?;
        let expected = [
            "Timestamp,Level,Target,Message,File,Line",
            "t1,INFO,purr_app::views::logs,plain,logs.rs,7",
            "t2,WARN,whisper,\"has, comma and \"\"quotes\"\"\nand a line break\",,",
            "",
        ]
        .join("\n");
        assert_eq!(csv, expected);
        Ok(())
    }

    #[test]
    fn json_round_trips_the_entries() -> miette::Result<()> {
        let entries = sample();
        let json = format_entries(LogExportFormat::Json, &entries, clock)?;
        let parsed: Vec<LogEntry> = serde_json::from_str(&json).into_diagnostic()?;
        let original: Vec<LogEntry> = entries.iter().map(|entry| (**entry).clone()).collect();
        assert_eq!(parsed, original);
        Ok(())
    }

    #[test]
    fn an_empty_selection_exports_an_empty_document() -> miette::Result<()> {
        let none: Vec<Arc<LogEntry>> = Vec::new();
        assert_eq!(format_entries(LogExportFormat::Text, &none, clock)?, "");
        assert_eq!(
            format_entries(LogExportFormat::Csv, &none, clock)?,
            "Timestamp,Level,Target,Message,File,Line\n"
        );
        assert_eq!(format_entries(LogExportFormat::Json, &none, clock)?, "[]");
        Ok(())
    }

    #[test]
    fn every_format_has_a_distinct_file_name() {
        let names: Vec<String> = LogExportFormat::ALL
            .iter()
            .map(|format| format.file_name())
            .collect();
        assert_eq!(
            names,
            vec!["purr-logs.log", "purr-logs.json", "purr-logs.csv"]
        );
    }
}
