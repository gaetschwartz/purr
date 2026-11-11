//! Logging infrastructure for capturing and displaying logs in the UI
//!
//! This module provides a custom tracing-subscriber layer that captures log records
//! in memory for real-time display in the logs viewer.

use miette::IntoDiagnostic as _;
use purr_common::platform::Platform as _;
use purr_common::settings::LogLevel;
use serde::{Deserialize, Serialize};
use serde_with::{DeserializeFromStr, SerializeDisplay};
use std::{
    borrow::Cow,
    collections::HashSet,
    fmt,
    sync::{
        OnceLock,
        atomic::{self},
    },
};
use tracing::{Level, Subscriber};
use tracing_subscriber::{
    EnvFilter, Registry,
    layer::{Context, Layer, Layered, SubscriberExt as _},
    registry::LookupSpan,
    reload,
};

use crate::platform::get_platform;

#[cfg(purr_web)]
fn now_millis() -> u64 {
    js_sys::Date::now() as u64
}

#[cfg(not(purr_web))]
fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis() as u64)
}

#[cfg(purr_web)]
fn local_clock_time(millis: u64) -> String {
    let date = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(millis as f64));
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        date.get_hours(),
        date.get_minutes(),
        date.get_seconds(),
        date.get_milliseconds()
    )
}

#[cfg(all(feature = "chrono", not(purr_web)))]
fn local_clock_time(millis: u64) -> String {
    i64::try_from(millis)
        .ok()
        .and_then(chrono::DateTime::from_timestamp_millis)
        .map(|utc| utc.with_timezone(&chrono::Local))
        .map_or_else(String::new, |local| {
            local.format("%H:%M:%S%.3f").to_string()
        })
}

#[cfg(not(any(feature = "chrono", purr_web)))]
fn local_clock_time(millis: u64) -> String {
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        (millis / 3_600_000) % 24,
        (millis / 60_000) % 60,
        (millis / 1000) % 60,
        millis % 1000
    )
}

/// Maximum number of log entries to keep in memory
const MAX_LOG_ENTRIES: usize = 10000;

/// A log entry captured by our custom layer
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
        local_clock_time(self.timestamp)
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
            Level::ERROR => Self::ERROR,
            Level::WARN => Self::WARN,
            Level::INFO => Self::INFO,
            Level::DEBUG => Self::DEBUG,
            Level::TRACE => Self::TRACE,
        }
    }
}

impl From<SerdeTracingLevel> for Level {
    fn from(level: SerdeTracingLevel) -> Self {
        match level {
            SerdeTracingLevel::ERROR => Self::ERROR,
            SerdeTracingLevel::WARN => Self::WARN,
            SerdeTracingLevel::INFO => Self::INFO,
            SerdeTracingLevel::DEBUG => Self::DEBUG,
            SerdeTracingLevel::TRACE => Self::TRACE,
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
                if let Some(search_query) = &query.search_query
                    && !entry.matches_search(search_query)
                {
                    return false;
                }
                if let Some(level_filters) = &query.level_filters
                    && !entry.matches_level(level_filters)
                {
                    return false;
                }
                if let Some(target_filters) = &query.target_filters
                    && !entry.matches_target(target_filters)
                {
                    return false;
                }
                true
            })
            .cloned();

        let mut entries: Vec<LogEntry> = filtered.collect();
        if let Some(limit) = query.limit {
            let excess = entries.len().saturating_sub(limit);
            entries.drain(..excess);
        }
        entries
    }

    /// Clear all log entries
    pub fn clear(&self) {
        while self.entries.pop().is_some() {}
    }

    /// Get unique targets from all entries
    pub fn get_unique_targets(&self) -> HashSet<String> {
        let guard = scc::Guard::new();

        self.entries
            .iter(&guard)
            .map(|entry| entry.target.clone())
            .collect::<HashSet<_>>()
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
            timestamp: now_millis(),
            level: SerdeTracingLevel::from(metadata.level()),
            target: metadata.target().to_owned(),
            message: String::new(),
            fields: Vec::new(),
            file: metadata.file().map(str::to_owned),
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
                .push((field.name().to_owned(), format!("{value:?}")));
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.entry.message = value.to_owned();
        } else {
            self.entry
                .fields
                .push((field.name().to_owned(), value.to_owned()));
        }
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.entry
            .fields
            .push((field.name().to_owned(), value.to_string()));
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.entry
            .fields
            .push((field.name().to_owned(), value.to_string()));
    }

    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        self.entry
            .fields
            .push((field.name().to_owned(), value.to_string()));
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.entry
            .fields
            .push((field.name().to_owned(), value.to_string()));
    }
}

type FilterHandle = reload::Handle<EnvFilter, Registry>;

static FILTER_HANDLE: OnceLock<FilterHandle> = OnceLock::new();

/// Initialize the logging capture layer
/// This should be called during application startup
pub fn init_logging_capture() -> miette::Result<()> {
    let filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy();
    let (filter_layer, handle) = reload::Layer::new(filter);
    FILTER_HANDLE
        .set(handle)
        .map_err(|_rejected| miette::miette!("Logging was already initialized"))?;

    let subscriber = Registry::default()
        .with(filter_layer)
        .with(output_layer())
        .with(LogsCaptureLayer::new());
    tracing::subscriber::set_global_default(subscriber).into_diagnostic()?;

    get_platform().install_logging_hooks();
    tracing::info!("Log capture initialized successfully");

    Ok(())
}

#[cfg(purr_web)]
fn output_layer() -> impl Layer<Layered<reload::Layer<EnvFilter, Registry>, Registry>> {
    purr_wasm::logging::console_layer()
}

#[cfg(not(purr_web))]
fn output_layer() -> impl Layer<Layered<reload::Layer<EnvFilter, Registry>, Registry>> {
    tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(true)
        .with_target(true)
        .with_file(true)
        .with_line_number(true)
        .compact()
}

/// Makes the application log at `level` and above.
pub fn apply_log_level(level: &LogLevel) -> miette::Result<()> {
    let handle = FILTER_HANDLE
        .get()
        .ok_or_else(|| miette::miette!("Logging has not been initialized"))?;
    handle
        .reload(EnvFilter::new(level.to_string()))
        .into_diagnostic()
}

/// Applies the log level chosen in the settings, unless `RUST_LOG` overrides it.
pub fn apply_saved_log_level(level: &LogLevel) {
    if std::env::var_os("RUST_LOG").is_some() {
        return;
    }
    if let Err(error) = apply_log_level(level) {
        tracing::warn!("Could not apply the saved log level: {error}");
    }
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

impl Default for LogsQuery<'_> {
    fn default() -> Self {
        Self::new()
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
            target: "test::module".to_owned(),
            message: "Test message".to_owned(),
            fields: vec![],
            file: Some("test.rs".to_owned()),
            line: Some(42),
        };

        storage.add_entry(entry);
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
            target: "test::module".to_owned(),
            message: "Important test message".to_owned(),
            fields: vec![("key".to_owned(), "value".to_owned())],
            file: Some("test.rs".to_owned()),
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
        assert!(entry.matches_target(&["test".to_owned()]));
        assert!(entry.matches_target(&["test::module".to_owned()]));
        assert!(!entry.matches_target(&["other".to_owned()]));
        assert!(entry.matches_target(&[])); // Empty filter matches all
    }
}
