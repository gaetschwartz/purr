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
    fmt,
    sync::{
        Arc, OnceLock,
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

/// In-memory storage for log entries, oldest first, bounded to the most recent entries.
#[derive(Debug)]
pub struct LogsStorage {
    entries: scc::Queue<Arc<LogEntry>>,
    len: atomic::AtomicUsize,
    capacity: usize,
    next_id: atomic::AtomicU64,
    revision: atomic::AtomicU64,
}

impl LogsStorage {
    /// Create a new logs storage instance
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(MAX_LOG_ENTRIES)
    }

    /// Create a storage that keeps at most `capacity` entries.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: scc::Queue::default(),
            len: atomic::AtomicUsize::new(0),
            capacity,
            next_id: atomic::AtomicU64::new(1),
            revision: atomic::AtomicU64::new(0),
        }
    }

    /// Adds an entry, numbering it and dropping the oldest ones beyond the capacity.
    pub fn add_entry(&self, mut entry: LogEntry) {
        entry.id = self.next_id.fetch_add(1, atomic::Ordering::Relaxed);

        self.len.fetch_add(1, atomic::Ordering::AcqRel);
        self.entries.push(Arc::new(entry));
        while self.len.load(atomic::Ordering::Acquire) > self.capacity {
            if self.entries.pop().is_none() {
                break;
            }
            self.len.fetch_sub(1, atomic::Ordering::AcqRel);
        }
        self.revision.fetch_add(1, atomic::Ordering::Release);
    }

    /// All entries, oldest first.
    pub fn snapshot(&self) -> Vec<Arc<LogEntry>> {
        let guard = scc::Guard::new();
        self.entries.iter(&guard).map(Arc::clone).collect()
    }

    /// A number that changes whenever entries are added or cleared.
    pub fn revision(&self) -> u64 {
        self.revision.load(atomic::Ordering::Acquire)
    }

    /// Removes every entry.
    pub fn clear(&self) {
        while self.entries.pop().is_some() {
            self.len.fetch_sub(1, atomic::Ordering::AcqRel);
        }
        self.revision.fetch_add(1, atomic::Ordering::Release);
    }

    /// Number of entries held.
    pub fn count(&self) -> usize {
        self.len.load(atomic::Ordering::Acquire)
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

/// Whether `RUST_LOG` decides the log level instead of the saved settings.
pub fn env_overrides_log_level() -> bool {
    std::env::var_os("RUST_LOG").is_some()
}

/// Applies the log level chosen in the settings, unless `RUST_LOG` overrides it.
pub fn apply_saved_log_level(level: &LogLevel) {
    if env_overrides_log_level() {
        return;
    }
    if let Err(error) = apply_log_level(level) {
        tracing::warn!("Could not apply the saved log level: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(message: &str) -> LogEntry {
        LogEntry {
            id: 0,
            timestamp: 1_234_567_890,
            level: SerdeTracingLevel::INFO,
            target: "test::module".to_owned(),
            message: message.to_owned(),
            fields: vec![],
            file: Some("test.rs".to_owned()),
            line: Some(42),
        }
    }

    fn messages(storage: &LogsStorage) -> Vec<String> {
        storage
            .snapshot()
            .iter()
            .map(|entry| entry.message.clone())
            .collect()
    }

    fn ids(storage: &LogsStorage) -> Vec<u64> {
        storage.snapshot().iter().map(|entry| entry.id).collect()
    }

    #[test]
    fn entries_are_kept_oldest_first_and_numbered_in_order() {
        let storage = LogsStorage::new();
        storage.add_entry(entry("first"));
        storage.add_entry(entry("second"));
        storage.add_entry(entry("third"));

        assert_eq!(ids(&storage), vec![1, 2, 3]);
        assert_eq!(messages(&storage), vec!["first", "second", "third"]);
        assert_eq!(storage.count(), 3);
    }

    #[test]
    fn the_oldest_entries_are_dropped_beyond_the_capacity() {
        let storage = LogsStorage::with_capacity(3);
        for message in ["a", "b", "c", "d", "e"] {
            storage.add_entry(entry(message));
        }

        assert_eq!(messages(&storage), vec!["c", "d", "e"]);
        assert_eq!(storage.count(), 3);
    }

    #[test]
    fn clearing_empties_the_storage_and_keeps_numbering() {
        let storage = LogsStorage::new();
        storage.add_entry(entry("before"));
        storage.clear();
        assert_eq!(storage.count(), 0);
        assert_eq!(messages(&storage), Vec::<String>::new());

        storage.add_entry(entry("after"));
        assert_eq!(ids(&storage), vec![2]);
    }

    #[test]
    fn the_revision_moves_on_every_change() {
        let storage = LogsStorage::new();
        let start = storage.revision();
        storage.add_entry(entry("one"));
        let after_add = storage.revision();
        storage.clear();
        let after_clear = storage.revision();

        assert_eq!(after_add, start + 1);
        assert_eq!(after_clear, after_add + 1);
    }

    #[test]
    fn a_snapshot_is_not_changed_by_later_entries() {
        let storage = LogsStorage::new();
        storage.add_entry(entry("one"));
        let snapshot = storage.snapshot();
        storage.add_entry(entry("two"));

        assert_eq!(snapshot.len(), 1);
        assert_eq!(storage.snapshot().len(), 2);
    }
}
