//! Logging infrastructure for capturing and displaying logs in the UI
//!
//! This module provides a custom tracing-subscriber layer that captures log records
//! in memory for real-time display in the logs viewer.

use miette::IntoDiagnostic as _;
use purr_common::platform::Platform as _;
use serde::{Deserialize, Serialize};
use serde_with::{DeserializeFromStr, SerializeDisplay};
use std::{
    borrow::Cow,
    collections::HashSet,
    fmt,
    sync::{
        atomic::{self},
        OnceLock,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tracing::{Level, Subscriber};
use tracing_subscriber::{
    layer::{Context, Layer},
    registry::LookupSpan,
};

use crate::platform::get_platform;

/// Maximum number of log entries to keep in memory
const MAX_LOG_ENTRIES: usize = 10000;

/// A log entry captured by our custom layer
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogEntry {
    /// Unique ID for the log entry
    pub id: u64,
    /// Timestamp when the log was created
    pub timestamp: u64,
    /// Log level (Error, Warn, Info, Debug, Trace)
    pub level: SerdeTracingLevel,
    /// Target/module that generated the log
    pub target: String,
    /// The log message content
    pub message: String,
    /// Optional fields associated with the log
    pub fields: Vec<(String, String)>,
    /// File location where the log was generated (if available)
    pub file: Option<String>,
    /// Line number where the log was generated (if available)
    pub line: Option<u32>,
}

impl LogEntry {
    /// Get timestamp as a formatted string
    #[must_use]
    pub fn formatted_timestamp(&self) -> String {
        #[cfg(feature = "chrono")]
        {
            let datetime = chrono::DateTime::from_timestamp(self.timestamp as i64 / 1000, 0)
                .unwrap_or_default();
            datetime.format("%H:%M:%S%.3f").to_string()
        }
        #[cfg(not(feature = "chrono"))]
        {
            // Fallback for web builds without chrono
            format!(
                "{:02}:{:02}:{:02}.{:03}",
                (self.timestamp / 3600000) % 24,
                (self.timestamp / 60000) % 60,
                (self.timestamp / 1000) % 60,
                self.timestamp % 1000
            )
        }
    }

    /// Get a short target name (last component of the module path)
    #[must_use]
    pub fn short_target(&self) -> &str {
        self.target.split("::").last().unwrap_or(&self.target)
    }

    /// Check if this log entry matches a search query
    #[must_use]
    pub fn matches_search(&self, query: &str) -> bool {
        if query.is_empty() {
            return true;
        }
        let query_lower = query.to_lowercase();
        self.message.to_lowercase().contains(&query_lower)
            || self.target.to_lowercase().contains(&query_lower)
            || self.fields.iter().any(|(k, v)| {
                k.to_lowercase().contains(&query_lower) || v.to_lowercase().contains(&query_lower)
            })
    }

    /// Check if this log entry matches a level filter
    #[must_use]
    pub fn matches_level(&self, filter_levels: &HashSet<SerdeTracingLevel>) -> bool {
        filter_levels.is_empty() || filter_levels.contains(&self.level)
    }

    /// Check if this log entry matches a target filter
    #[must_use]
    pub fn matches_target(&self, filter_targets: &[String]) -> bool {
        filter_targets.is_empty()
            || filter_targets
                .iter()
                .any(|target| self.target.starts_with(target))
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    SerializeDisplay,
    DeserializeFromStr,
    PartialEq,
    Eq,
    Hash,
    strum::EnumString,
    strum::Display,
    strum::VariantArray,
)]
pub enum SerdeTracingLevel {
    ERROR,
    WARN,
    INFO,
    DEBUG,
    TRACE,
}

impl From<&Level> for SerdeTracingLevel {
    fn from(level: &Level) -> Self {
        match *level {
            Level::ERROR => SerdeTracingLevel::ERROR,
            Level::WARN => SerdeTracingLevel::WARN,
            Level::INFO => SerdeTracingLevel::INFO,
            Level::DEBUG => SerdeTracingLevel::DEBUG,
            Level::TRACE => SerdeTracingLevel::TRACE,
        }
    }
}

impl From<SerdeTracingLevel> for Level {
    fn from(level: SerdeTracingLevel) -> Self {
        match level {
            SerdeTracingLevel::ERROR => Level::ERROR,
            SerdeTracingLevel::WARN => Level::WARN,
            SerdeTracingLevel::INFO => Level::INFO,
            SerdeTracingLevel::DEBUG => Level::DEBUG,
            SerdeTracingLevel::TRACE => Level::TRACE,
        }
    }
}

/// In-memory storage for log entries
#[derive(Debug)]
pub struct LogsStorage {
    entries: scc::Queue<LogEntry>,
    next_id: atomic::AtomicU64,
}

impl LogsStorage {
    /// Create a new logs storage instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: scc::Queue::default(),
            next_id: atomic::AtomicU64::new(1),
        }
    }

    /// Add a new log entry to storage
    pub fn add_entry(&self, mut entry: LogEntry) {
        // Assign a unique ID based on current timestamp and a random component
        entry.id = self.next_id.fetch_add(1, atomic::Ordering::Relaxed);

        self.entries.push(entry);

        // Maintain maximum size
        while self.entries.len() > MAX_LOG_ENTRIES {
            self.entries.pop();
        }
    }

    /// Get all log entries
    pub fn get_entries(&self) -> Vec<LogEntry> {
        let guard = scc::Guard::new();
        self.entries.iter(&guard).cloned().collect()
    }

    /// Get filtered log entries
    pub fn get_filtered_entries(&self, query: LogsQuery) -> Vec<LogEntry> {
        let guard = scc::Guard::new();
        let filtered = self
            .entries
            .iter(&guard)
            .filter(|entry| {
                if let Some(search_query) = &query.search_query {
                    if !entry.matches_search(search_query) {
                        return false;
                    }
                }
                if let Some(level_filters) = &query.level_filters {
                    if !entry.matches_level(level_filters) {
                        return false;
                    }
                }
                if let Some(target_filters) = &query.target_filters {
                    if !entry.matches_target(target_filters) {
                        return false;
                    }
                }
                true
            })
            .cloned();

        if let Some(limit) = query.limit {
            filtered.take(limit).collect()
        } else {
            filtered.collect()
        }
    }

    /// Clear all log entries
    pub fn clear(&self) {
        while self.entries.pop().is_some() {}
    }

    /// Get unique targets from all entries
    pub fn get_unique_targets(&self) -> HashSet<String> {
        let guard = scc::Guard::new();
        let targets = self
            .entries
            .iter(&guard)
            .map(|entry| entry.target.clone())
            .collect::<HashSet<_>>();
        targets
    }

    /// Get entry count
    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

impl Default for LogsStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// Global instance of logs storage
static LOGS_STORAGE: OnceLock<LogsStorage> = OnceLock::new();

/// Get the global logs storage instance
pub fn get_logs_storage() -> &'static LogsStorage {
    LOGS_STORAGE.get_or_init(LogsStorage::new)
}

/// Custom tracing layer that captures logs for the UI
pub struct LogsCaptureLayer {}

impl LogsCaptureLayer {
    /// Create a new logs capture layer
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for LogsCaptureLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Layer<S> for LogsCaptureLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();

        // Create a log entry
        let mut entry = LogEntry {
            id: 0, // Will be set when adding to storage
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            level: SerdeTracingLevel::from(metadata.level()),
            target: metadata.target().to_string(),
            message: String::new(),
            fields: Vec::new(),
            file: metadata.file().map(std::string::ToString::to_string),
            line: metadata.line(),
        };

        // Capture the message and fields using a custom visitor
        let mut visitor = LogFieldVisitor::new(&mut entry);
        event.record(&mut visitor);

        // Store the entry
        get_logs_storage().add_entry(entry);
    }
}

/// Visitor to extract fields from tracing events
struct LogFieldVisitor<'a> {
    entry: &'a mut LogEntry,
}

impl<'a> LogFieldVisitor<'a> {
    fn new(entry: &'a mut LogEntry) -> Self {
        Self { entry }
    }
}

impl tracing::field::Visit for LogFieldVisitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            self.entry.message = format!("{value:?}");
            // Remove quotes around the message if they exist
            if self.entry.message.starts_with('"') && self.entry.message.ends_with('"') {
                self.entry.message =
                    self.entry.message[1..self.entry.message.len() - 1].to_string();
            }
        } else {
            self.entry
                .fields
                .push((field.name().to_string(), format!("{value:?}")));
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.entry.message = value.to_string();
        } else {
            self.entry
                .fields
                .push((field.name().to_string(), value.to_string()));
        }
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.entry
            .fields
            .push((field.name().to_string(), value.to_string()));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.entry
            .fields
            .push((field.name().to_string(), value.to_string()));
    }

    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.entry
            .fields
            .push((field.name().to_string(), value.to_string()));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.entry
            .fields
            .push((field.name().to_string(), value.to_string()));
    }
}

/// Initialize the logging capture layer
/// This should be called during application startup
pub fn init_logging_capture() -> miette::Result<()> {
    use tracing_subscriber::{prelude::*, EnvFilter, Registry};

    // Create the logs capture layer
    let logs_layer = LogsCaptureLayer::new();

    // Create a filter that allows all levels by default, but respects RUST_LOG
    let filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy();

    // Build the subscriber with both console output and our capture layer
    let subscriber = Registry::default()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_ansi(true)
                .with_target(true)
                .with_file(true)
                .with_line_number(true)
                .compact(),
        )
        .with(logs_layer);

    tracing::subscriber::set_global_default(subscriber).into_diagnostic()?;

    get_platform().install_logging_hooks();
    // Log that initialization is complete
    tracing::info!("Log capture initialized successfully");

    Ok(())
}

/// Export logs to a string in various formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogExportFormat {
    Text,
    Json,
    Csv,
}

impl LogsStorage {
    /// Export logs to a string in the specified format
    pub fn export_logs(&self, format: LogExportFormat, query: LogsQuery) -> String {
        let entries = self.get_filtered_entries(query);

        match format {
            LogExportFormat::Text => entries
                .iter()
                .map(|entry| {
                    format!(
                        "{} [{:5}] {}: {}",
                        entry.formatted_timestamp(),
                        entry.level,
                        entry.short_target(),
                        entry.message
                    )
                })
                .collect::<Vec<_>>()
                .join("\n"),
            LogExportFormat::Json => serde_json::to_string_pretty(&entries).unwrap_or_default(),
            LogExportFormat::Csv => {
                let mut csv = String::new();
                csv.push_str("Timestamp,Level,Target,Message,File,Line\n");
                for entry in entries {
                    csv.push_str(&format!(
                        "{},{},{},\"{}\",{},{}\n",
                        entry.formatted_timestamp(),
                        entry.level,
                        entry.target,
                        entry.message.replace('"', "\"\""), // Escape quotes in CSV
                        entry.file.unwrap_or_default(),
                        entry.line.unwrap_or_default()
                    ));
                }
                csv
            }
        }
    }
}

pub struct LogsQuery<'a> {
    search_query: Option<String>,
    level_filters: Option<Cow<'a, HashSet<SerdeTracingLevel>>>,
    target_filters: Option<Vec<String>>,
    limit: Option<usize>,
}

impl<'a> LogsQuery<'a> {
    pub fn new() -> Self {
        Self {
            search_query: None,
            level_filters: None,
            target_filters: None,
            limit: None,
        }
    }

    pub fn with_search(mut self, query: impl Into<String>) -> Self {
        self.search_query = Some(query.into());
        self
    }

    pub fn with_level_filters(
        mut self,
        levels: impl Into<Cow<'a, HashSet<SerdeTracingLevel>>>,
    ) -> Self {
        self.level_filters = Some(levels.into());
        self
    }

    pub fn with_target_filters(mut self, targets: Vec<String>) -> Self {
        self.target_filters = Some(targets);
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn query(&self) -> Option<&str> {
        self.search_query.as_deref()
    }

    pub fn level_filters(&self) -> Option<&HashSet<SerdeTracingLevel>> {
        self.level_filters.as_deref()
    }

    pub fn target_filters(&self) -> Option<&[String]> {
        self.target_filters.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logs_storage_basic_operations() {
        let storage = LogsStorage::new();

        let entry = LogEntry {
            id: 0,
            timestamp: 1234567890,
            level: SerdeTracingLevel::INFO,
            target: "test::module".to_string(),
            message: "Test message".to_string(),
            fields: vec![],
            file: Some("test.rs".to_string()),
            line: Some(42),
        };

        storage.add_entry(entry.clone());
        assert_eq!(storage.count(), 1);

        let entries = storage.get_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].message, "Test message");
    }

    #[test]
    fn test_log_entry_filtering() {
        let entry = LogEntry {
            id: 1,
            timestamp: 1234567890,
            level: SerdeTracingLevel::INFO,
            target: "test::module".to_string(),
            message: "Important test message".to_string(),
            fields: vec![("key".to_string(), "value".to_string())],
            file: Some("test.rs".to_string()),
            line: Some(42),
        };

        // Test search filtering
        assert!(entry.matches_search("test"));
        assert!(entry.matches_search("Important"));
        assert!(entry.matches_search("value"));
        assert!(!entry.matches_search("nonexistent"));

        // Test level filtering
        assert!(entry.matches_level(&[SerdeTracingLevel::INFO].into_iter().collect()));
        assert!(!entry.matches_level(&[SerdeTracingLevel::ERROR].into_iter().collect()));
        assert!(entry.matches_level(&[].into_iter().collect())); // Empty filter matches all

        // Test target filtering
        assert!(entry.matches_target(&["test".to_string()]));
        assert!(entry.matches_target(&["test::module".to_string()]));
        assert!(!entry.matches_target(&["other".to_string()]));
        assert!(entry.matches_target(&[])); // Empty filter matches all
    }
}
