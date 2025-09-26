//! Logs viewer component for displaying application logs in real-time

use crate::platform::logging::{get_logs_storage, LogEntry, LogExportFormat};
use dioxus::prelude::*;
use std::collections::HashSet;

// Use LogEntry from the platform logging module

/// Get logs from the storage system
fn get_logs(
    search_query: &str,
    selected_levels: &HashSet<String>,
    max_display: usize,
) -> Vec<LogEntry> {
    let storage = get_logs_storage();
    let level_filters: Vec<String> = selected_levels.iter().cloned().collect();

    storage.get_filtered_entries(
        search_query,
        &level_filters,
        &[], // No target filters for now
        Some(max_display),
    )
}

/// Generate some example logs for testing
fn generate_example_logs() {
    use std::thread;
    use std::time::Duration;

    // Spawn a background task to generate example logs
    thread::spawn(|| {
        let mut counter = 0;
        loop {
            counter += 1;

            match counter % 4 {
                0 => tracing::info!("Example info log #{}", counter),
                1 => tracing::warn!(duration_ms = 150, "Example warning log #{}", counter),
                2 => tracing::debug!(component = "ui", "Example debug log #{}", counter),
                3 => tracing::error!(error_code = 500, "Example error log #{}", counter),
                _ => {},
            }

            thread::sleep(Duration::from_secs(2));

            // Stop after 20 logs to prevent infinite spam
            if counter >= 20 {
                break;
            }
        }
    });
}

/// Available log levels for filtering
const LOG_LEVELS: &[&str] = &["ERROR", "WARN", "INFO", "DEBUG", "TRACE"];

/// Log viewer settings and state
#[derive(Clone, PartialEq)]
struct LogsState {
    /// Search query for filtering logs
    search_query: String,
    /// Selected log levels for filtering
    selected_levels: HashSet<String>,
    /// Whether auto-refresh is enabled
    auto_refresh: bool,
    /// Whether to stick to bottom when new logs arrive
    stick_to_bottom: bool,
    /// Maximum number of logs to display
    max_display: usize,
    /// Whether the logs are paused
    paused: bool,
}

impl Default for LogsState {
    fn default() -> Self {
        Self {
            search_query: String::new(),
            selected_levels: HashSet::new(), // Empty means all levels
            auto_refresh: true,
            stick_to_bottom: true,
            max_display: 1000,
            paused: false,
        }
    }
}

/// Main logs viewer component
#[component]
pub fn Logs() -> Element {
    let mut state = use_signal(LogsState::default);
    let mut logs = use_signal(Vec::<LogEntry>::new);

    // Initial load and setup
    use_effect(move || {
        // Generate some example logs for testing
        generate_example_logs();

        // Load initial logs
        let initial_logs = get_logs("", &HashSet::new(), 1000);
        logs.set(initial_logs);
    });

    // Real-time updates using a simple timer
    let state_clone = state.clone();
    let mut logs_clone = logs.clone();

    use_effect(move || {
        // Create a simple interval for updates
        spawn(async move {
            loop {
                // Wait for 1 second
                #[cfg(target_arch = "wasm32")]
                {
                    use gloo_timers::future::TimeoutFuture;
                    TimeoutFuture::new(1000).await;
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                }

                let current_state = state_clone.read();
                if !current_state.paused && current_state.auto_refresh {
                    let updated_logs = get_logs(
                        &current_state.search_query,
                        &current_state.selected_levels,
                        current_state.max_display,
                    );
                    logs_clone.set(updated_logs);
                }
            }
        });
    });

    let current_state = state.read();
    let filtered_logs = logs.read().clone();

    rsx! {
        div { class: "flex flex-col h-screen bg-gray-50 dark:bg-gray-900",

            // Header with controls
            header { class: "bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4",

                div { class: "flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4",

                    // Title and stats
                    div { class: "flex items-center gap-4",
                        h1 { class: "text-2xl font-bold text-gray-900 dark:text-white",
                            "Application Logs"
                        }

                        div { class: "flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",
                            span { class: "inline-flex items-center px-2 py-1 rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
                                {format!("{} entries", filtered_logs.len())}
                            }

                            if current_state.paused {
                                span { class: "inline-flex items-center px-2 py-1 rounded-full bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
                                    "⏸ PAUSED"
                                }
                            }
                        }
                    }

                    // Controls
                    div { class: "flex flex-wrap items-center gap-3",

                        // Search input
                        div { class: "relative",
                            input {
                                r#type: "text",
                                placeholder: "Search logs...",
                                class: "w-64 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-transparent",
                                value: current_state.search_query.clone(),
                                oninput: move |e| {
                                    state.write().search_query = e.value();
                                },
                            }
                        }

                        // Level filters
                        div { class: "flex items-center gap-2",
                            span { class: "text-sm font-medium text-gray-700 dark:text-gray-300",
                                "Levels:"
                            }

                            for level in LOG_LEVELS {
                                LogLevelButton {
                                    level: level.to_string(),
                                    selected: current_state.selected_levels.contains(*level),
                                    on_toggle: move |level: String| {
                                        let mut current_state = state.write();
                                        if current_state.selected_levels.contains(&level) {
                                            current_state.selected_levels.remove(&level);
                                        } else {
                                            current_state.selected_levels.insert(level);
                                        }
                                    },
                                }
                            }
                        }

                        // Action buttons
                        div { class: "flex items-center gap-2",

                            // Pause/Resume
                            button {
                                class: if current_state.paused { "px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md transition-colors duration-200" } else { "px-3 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-md transition-colors duration-200" },
                                onclick: move |_| {
                                    let current_paused = state.read().paused;
                                    state.write().paused = !current_paused;
                                },
                                if current_state.paused {
                                    "▶ Resume"
                                } else {
                                    "⏸ Pause"
                                }
                            }

                            // Clear logs
                            button {
                                class: "px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md transition-colors duration-200",
                                onclick: move |_| {
                                    // Clear the global logs storage
                                    get_logs_storage().clear();
                                    // Clear the local state too
                                    logs.set(vec![]);
                                },
                                "🗑 Clear"
                            }

                            // Export dropdown
                            ExportDropdown { state: state }
                        }
                    }
                }
            }

            // Filters bar (if any filters are active)
            if !current_state.selected_levels.is_empty() {
                div { class: "bg-blue-50 dark:bg-blue-900/20 border-b border-blue-200 dark:border-blue-800 px-6 py-2",
                    div { class: "flex items-center gap-4 text-sm",
                        span { class: "font-medium text-blue-900 dark:text-blue-200",
                            "Active filters:"
                        }

                        div { class: "flex items-center gap-2",
                            span { class: "text-blue-700 dark:text-blue-300", "Levels:" }
                            for level in &current_state.selected_levels {
                                span { class: "px-2 py-1 bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-200 rounded text-xs",
                                    {level.clone()}
                                }
                            }
                        }

                        button {
                            class: "text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-200 font-medium",
                            onclick: move |_| {
                                state.write().selected_levels.clear();
                            },
                            "Clear filters"
                        }
                    }
                }
            }

            // Main logs display
            main { class: "flex-1 overflow-hidden",

                div {
                    class: "h-full overflow-y-auto font-mono text-sm bg-white dark:bg-gray-900",
                    id: "logs-container",

                    if filtered_logs.is_empty() {
                        div { class: "flex items-center justify-center h-full text-gray-500 dark:text-gray-400",
                            div { class: "text-center",
                                div { class: "text-6xl mb-4", "📝" }
                                h3 { class: "text-xl font-semibold mb-2", "No logs found" }
                                p {
                                    "Try adjusting your search or filter criteria, or wait for new logs to appear"
                                }
                            }
                        }
                    } else {
                        div { class: "divide-y divide-gray-100 dark:divide-gray-800",

                            for (index , entry) in filtered_logs.iter().enumerate() {
                                LogEntryRow {
                                    key: "{entry.id}-{index}",
                                    entry: entry.clone(),
                                    search_query: current_state.search_query.clone(),
                                }
                            }
                        }
                    }
                }
            }

            // Footer with settings
            footer { class: "bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-6 py-3",
                div { class: "flex items-center justify-between text-sm text-gray-600 dark:text-gray-400",

                    div { class: "flex items-center gap-4",

                        label { class: "flex items-center gap-2 cursor-pointer",
                            input {
                                r#type: "checkbox",
                                checked: current_state.auto_refresh,
                                class: "rounded border-gray-300 dark:border-gray-600",
                                onchange: move |e| {
                                    state.write().auto_refresh = e.checked();
                                },
                            }
                            "Auto-refresh"
                        }

                        label { class: "flex items-center gap-2 cursor-pointer",
                            input {
                                r#type: "checkbox",
                                checked: current_state.stick_to_bottom,
                                class: "rounded border-gray-300 dark:border-gray-600",
                                onchange: move |e| {
                                    state.write().stick_to_bottom = e.checked();
                                },
                            }
                            "Stick to bottom"
                        }
                    }

                    div {
                        {format!("Displaying {} logs (total: {})", filtered_logs.len(), get_logs_storage().count())}
                    }
                }
            }
        }
    }
}

/// Individual log entry row component
#[component]
fn LogEntryRow(entry: LogEntry, search_query: String) -> Element {
    let level_indicator = match entry.level.as_str() {
        "ERROR" => "🔴",
        "WARN" => "🟡",
        "INFO" => "🔵",
        "DEBUG" => "⚪",
        "TRACE" => "⚫",
        _ => "⚪",
    };

    rsx! {
        div { class: "px-6 py-3 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors",

            div { class: "flex items-start gap-3",

                // Level indicator
                div { class: "flex-shrink-0 mt-0.5",
                    span { class: "text-lg", {level_indicator} }
                }

                // Content
                div { class: "flex-1 min-w-0",

                    // Header line with timestamp, level, and target
                    div { class: "flex items-center gap-3 mb-1 text-xs",

                        time { class: "font-mono text-gray-500 dark:text-gray-400 flex-shrink-0",
                            {entry.formatted_timestamp()}
                        }

                        span {
                            class: format!(
                                "px-2 py-1 rounded text-xs font-semibold {}",
                                match entry.level.as_str() {
                                    "ERROR" => "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
                                    "WARN" => {
                                        "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200"
                                    }
                                    "INFO" => "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
                                    "DEBUG" => "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200",
                                    "TRACE" => "bg-gray-50 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
                                    _ => "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200",
                                },
                            ),
                            {entry.level.clone()}
                        }

                        code { class: "text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded text-xs",
                            {entry.short_target()}
                        }

                        if let Some(file) = &entry.file {
                            span { class: "text-gray-400 dark:text-gray-600 text-xs",
                                {format!("{}:{}", file, entry.line.unwrap_or(0))}
                            }
                        }
                    }

                    // Message
                    div { class: "text-gray-900 dark:text-white leading-relaxed",
                        LogMessage {
                            message: entry.message.clone(),
                            search_query: search_query.clone(),
                        }
                    }

                    // Fields (if any)
                    if !entry.fields.is_empty() {
                        div { class: "mt-2 space-y-1",
                            for (key , value) in &entry.fields {
                                div { class: "text-xs text-gray-600 dark:text-gray-400",
                                    span { class: "font-medium", {key.clone()} }
                                    ": "
                                    code { class: "bg-gray-100 dark:bg-gray-800 px-1 rounded",
                                        {value.clone()}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Log message component with search highlighting
#[component]
fn LogMessage(message: String, search_query: String) -> Element {
    if search_query.is_empty() {
        return rsx! {
            {message}
        };
    }

    let query_lower = search_query.to_lowercase();
    let message_lower = message.to_lowercase();

    if let Some(start) = message_lower.find(&query_lower) {
        let end = start + search_query.len();
        let before = &message[..start];
        let matched = &message[start..end];
        let after = &message[end..];

        rsx! {
            {before}
            mark { class: "bg-yellow-200 dark:bg-yellow-600 px-1 rounded", {matched} }
            LogMessage { message: after.to_string(), search_query }
        }
    } else {
        rsx! {
            {message}
        }
    }
}

/// Log level filter button
#[component]
fn LogLevelButton(level: String, selected: bool, on_toggle: EventHandler<String>) -> Element {
    let (bg_class, text_class) = if selected {
        match level.as_str() {
            "ERROR" => ("bg-red-600 hover:bg-red-700", "text-white"),
            "WARN" => ("bg-yellow-600 hover:bg-yellow-700", "text-white"),
            "INFO" => ("bg-blue-600 hover:bg-blue-700", "text-white"),
            "DEBUG" => ("bg-gray-600 hover:bg-gray-700", "text-white"),
            "TRACE" => ("bg-gray-500 hover:bg-gray-600", "text-white"),
            _ => ("bg-gray-600 hover:bg-gray-700", "text-white"),
        }
    } else {
        ("bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 border border-gray-300 dark:border-gray-600", "text-gray-700 dark:text-gray-300")
    };

    let level_text = level.clone();

    rsx! {
        button {
            class: format!(
                "px-2 py-1 text-xs rounded transition-colors duration-200 {} {}",
                bg_class,
                text_class,
            ),
            onclick: move |_| on_toggle.call(level.clone()),
            {level_text}
        }
    }
}

/// Export dropdown component
#[component]
fn ExportDropdown(state: Signal<LogsState>) -> Element {
    let mut show_dropdown = use_signal(|| false);

    rsx! {
        div { class: "relative",

            button {
                class: "px-3 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-md transition-colors duration-200 flex items-center gap-2",
                onclick: move |_| {
                    let current = *show_dropdown.read();
                    show_dropdown.set(!current);
                },
                "💾 Export"
                span { class: if *show_dropdown.read() { "transform rotate-180 transition-transform" } else { "transition-transform" },
                    "▼"
                }
            }

            if *show_dropdown.read() {
                div {
                    class: "absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg z-50",
                    onclick: move |_| show_dropdown.set(false), // Close on click

                    div { class: "py-1",

                        button {
                            class: "block w-full px-4 py-2 text-left text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors",
                            onclick: move |_| {
                                export_logs(LogExportFormat::Text, &state);
                                show_dropdown.set(false);
                            },
                            "📄 Export as Text"
                        }

                        button {
                            class: "block w-full px-4 py-2 text-left text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors",
                            onclick: move |_| {
                                export_logs(LogExportFormat::Json, &state);
                                show_dropdown.set(false);
                            },
                            "📋 Export as JSON"
                        }

                        button {
                            class: "block w-full px-4 py-2 text-left text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors",
                            onclick: move |_| {
                                export_logs(LogExportFormat::Csv, &state);
                                show_dropdown.set(false);
                            },
                            "📊 Export as CSV"
                        }
                    }
                }
            }
        }
    }
}

/// Export logs to a downloadable file
fn export_logs(format: LogExportFormat, state: &Signal<LogsState>) {
    let current_state = state.read();
    let storage = get_logs_storage();
    let level_filters: Vec<String> = current_state.selected_levels.iter().cloned().collect();

    let exported_data = storage.export_logs(
        format,
        &current_state.search_query,
        &level_filters,
        &[], // No target filters for now
    );

    let (filename, _content_type) = match format {
        LogExportFormat::Text => ("logs.txt", "text/plain"),
        LogExportFormat::Json => ("logs.json", "application/json"),
        LogExportFormat::Csv => ("logs.csv", "text/csv"),
    };

    // Use platform-specific export
    #[cfg(feature = "desktop")]
    {
        spawn(async move {
            if let Err(e) = save_file_desktop(filename, &exported_data).await {
                tracing::error!("Failed to save file: {}", e);
            }
        });
    }

    #[cfg(all(feature = "web", target_arch = "wasm32"))]
    {
        if let Err(e) = save_file_web(filename, content_type, &exported_data) {
            tracing::error!("Failed to download file: {}", e);
        }
    }

    tracing::info!("Exported {} logs as {}", storage.count(), filename);
}

/// Save file on desktop platforms
#[cfg(feature = "desktop")]
async fn save_file_desktop(filename: &str, content: &str) -> Result<(), Box<dyn std::error::Error>> {
    use rfd::AsyncFileDialog;
    use std::io::Write;

    if let Some(file_handle) = AsyncFileDialog::new()
        .set_file_name(filename)
        .save_file()
        .await
    {
        let mut file = std::fs::File::create(file_handle.path())?;
        file.write_all(content.as_bytes())?;
        tracing::info!("File saved to: {}", file_handle.path().display());
    }

    Ok(())
}

/// Save file on web platforms
#[cfg(all(feature = "web", target_arch = "wasm32"))]
fn save_file_web(filename: &str, content_type: &str, content: &str) -> Result<(), Box<dyn std::error::Error>> {
    use web_sys::{window, Blob, BlobPropertyBag, Url, Document, HtmlElement};
    use wasm_bindgen::JsCast;

    let window = window().ok_or("No window object")?;
    let document = window.document().ok_or("No document object")?;

    // Create blob
    let mut blob_options = BlobPropertyBag::new();
    blob_options.type_(content_type);

    let array = js_sys::Array::new();
    array.push(&content.into());

    let blob = Blob::new_with_str_sequence_and_options(&array, &blob_options)?;
    let url = Url::create_object_url_with_blob(&blob)?;

    // Create download link
    let link = document.create_element("a")?;
    let html_link = link.dyn_into::<HtmlElement>()?;

    html_link.set_attribute("href", &url)?;
    html_link.set_attribute("download", filename)?;
    html_link.click();

    // Clean up
    Url::revoke_object_url(&url)?;

    Ok(())
}
