//! Logs viewer component for displaying application logs in real-time

use crate::{
    components::{
        button::Button,
        icon::{Icon, IconName, IconSize},
        layer::PopoverAlign,
        page::Page,
        popover::{Popover, PopoverOption},
        toast::{Toasts, use_toasts},
    },
    platform::{
        export::{TextFile, save_text_file},
        logging::{LogEntry, LogExportFormat, LogsQuery, SerdeTracingLevel, get_logs_storage},
    },
};
use dioxus::{core::spawn_forever, prelude::*};
use std::{borrow::Cow, collections::HashSet};
use strum::VariantArray as _;

// Use LogEntry from the platform logging module

/// Log viewer settings and state
#[derive(Clone, PartialEq)]
struct LogsState {
    /// Search query for filtering logs
    search_query: String,
    /// Selected log levels for filtering
    selected_levels: HashSet<SerdeTracingLevel>,
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
        // Load initial logs
        let initial_logs =
            get_logs_storage().get_filtered_entries(LogsQuery::new().with_limit(1000));
        logs.set(initial_logs);
    });

    // Real-time updates using a simple timer
    let state_clone = state;
    let mut logs_clone = logs;

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
                    let updated_logs = get_logs_storage().get_filtered_entries(
                        LogsQuery::new()
                            .with_search(&current_state.search_query)
                            .with_level_filters(Cow::Borrowed(&current_state.selected_levels))
                            .with_limit(current_state.max_display),
                    );
                    logs_clone.set(updated_logs);
                    if current_state.stick_to_bottom {
                        document::eval(
                            "const box = document.getElementById('logs-container'); if (box) { box.scrollTop = box.scrollHeight; }",
                        );
                    }
                }
            }
        });
    });

    let current_state = state.read();
    let filtered_logs = logs.read().clone();

    rsx! {
        Page { title: "Logs", subtitle: "Logs stay on this device.", wide: true,
        section { class: "card glass tight", aria_label: "Log entries",

            // Header with controls
            div { class: "logbar",

                div { class: "grow flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4",

                    // Title and stats
                    div { class: "flex items-center gap-4",
                        div { class: "flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400",
                            span { class: "inline-flex items-center px-2 py-1 rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
                                {if filtered_logs.len() == 1 { "1 entry".to_owned() } else { format!("{} entries", filtered_logs.len()) }}
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

                            for & level in SerdeTracingLevel::VARIANTS {
                                LogLevelButton {
                                    level,
                                    selected: current_state.selected_levels.contains(&level),
                                    on_toggle: move |level: SerdeTracingLevel| {
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
                                    get_logs_storage().clear();
                                    logs.set(vec![]);
                                },
                                "🗑 Clear"
                            }

                            // Export dropdown
                            ExportDropdown { state }
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
                                    {level.to_string()}
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
            div {

                div {
                    class: "loglist",
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
            div { class: "logbar",
                div { class: "grow flex flex-wrap gap-2 items-center justify-between text-sm text-gray-600 dark:text-gray-400",

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
                        {
                            format!(
                                "Displaying {} logs (total: {})",
                                filtered_logs.len(),
                                get_logs_storage().count(),
                            )
                        }
                    }
                }
            }
        }
        }
    }
}

/// Individual log entry row component
#[component]
fn LogEntryRow(entry: LogEntry, search_query: String) -> Element {
    let level_indicator = match entry.level {
        SerdeTracingLevel::ERROR => "🔴",
        SerdeTracingLevel::WARN => "🟡",
        SerdeTracingLevel::INFO => "🔵",
        SerdeTracingLevel::DEBUG => "⚪",
        SerdeTracingLevel::TRACE => "⚫",
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
                    div { class: "flex flex-wrap items-center gap-3 mb-1 text-xs",

                        time { class: "font-mono text-gray-500 dark:text-gray-400 flex-shrink-0",
                            {entry.formatted_timestamp()}
                        }

                        span {
                            class: format!(
                                "px-2 py-1 rounded text-xs font-semibold {}",
                                match entry.level {
                                    SerdeTracingLevel::ERROR => {
                                        "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                                    }
                                    SerdeTracingLevel::WARN => {
                                        "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200"
                                    }
                                    SerdeTracingLevel::INFO => {
                                        "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
                                    }
                                    SerdeTracingLevel::DEBUG => {
                                        "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200"
                                    }
                                    SerdeTracingLevel::TRACE => {
                                        "bg-gray-50 text-gray-600 dark:bg-gray-800 dark:text-gray-400"
                                    }
                                },
                            ),
                            {entry.level.to_string()}
                        }

                        code { class: "text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded text-xs",
                            {entry.short_target()}
                        }

                        if let Some(file) = &entry.file {
                            span { class: "text-gray-400 dark:text-gray-600 text-xs break-all",
                                {format!("{}:{}", file, entry.line.unwrap_or(0))}
                            }
                        }
                    }

                    // Message
                    div { class: "text-gray-900 dark:text-white leading-relaxed",
                        LogMessage {
                            message: entry.message.clone(),
                            search_query: search_query,
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
            LogMessage { message: after.to_owned(), search_query }
        }
    } else {
        rsx! {
            {message}
        }
    }
}

/// Log level filter button
#[component]
fn LogLevelButton(
    level: SerdeTracingLevel,
    selected: bool,
    on_toggle: EventHandler<SerdeTracingLevel>,
) -> Element {
    let (bg_class, text_class) = if selected {
        match level {
            SerdeTracingLevel::ERROR => ("bg-red-600 hover:bg-red-700", "text-white"),
            SerdeTracingLevel::WARN => ("bg-yellow-600 hover:bg-yellow-700", "text-white"),
            SerdeTracingLevel::INFO => ("bg-blue-600 hover:bg-blue-700", "text-white"),
            SerdeTracingLevel::DEBUG => ("bg-gray-600 hover:bg-gray-700", "text-white"),
            SerdeTracingLevel::TRACE => ("bg-gray-500 hover:bg-gray-600", "text-white"),
        }
    } else {
        (
            "bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 border border-gray-300 dark:border-gray-600",
            "text-gray-700 dark:text-gray-300",
        )
    };

    let level_text = level.to_string();

    rsx! {
        button {
            class: format!(
                "px-2 py-1 text-xs rounded transition-colors duration-200 {} {}",
                bg_class,
                text_class,
            ),
            onclick: move |_| on_toggle.call(level),
            {level_text}
        }
    }
}

/// Export menu: text, JSON or CSV of the entries on screen
#[component]
fn ExportDropdown(state: Signal<LogsState>) -> Element {
    let mut open = use_signal(|| false);
    let toasts = use_toasts();
    let mut export = move |format: LogExportFormat| {
        open.set(false);
        export_logs(format, &state, toasts);
    };

    rsx! {
        Button {
            id: "log-export",
            icon: IconName::Share,
            aria_haspopup: "menu",
            aria_expanded: open(),
            onclick: move |_| open.toggle(),
            "Export"
        }
        if open() {
            Popover {
                anchor: "log-export",
                label: "Export logs",
                menu: true,
                align: PopoverAlign::End,
                on_close: move |()| open.set(false),
                div { class: "plist", role: "menu", aria_label: "Export logs",
                    PopoverOption {
                        title: "Text",
                        trailing: ".txt",
                        leading: rsx! { Icon { name: IconName::Doc, size: IconSize::Small } },
                        onselect: move |()| export(LogExportFormat::Text),
                    }
                    PopoverOption {
                        title: "JSON",
                        trailing: ".json",
                        leading: rsx! { Icon { name: IconName::Doc, size: IconSize::Small } },
                        onselect: move |()| export(LogExportFormat::Json),
                    }
                    PopoverOption {
                        title: "CSV",
                        trailing: ".csv",
                        leading: rsx! { Icon { name: IconName::Doc, size: IconSize::Small } },
                        onselect: move |()| export(LogExportFormat::Csv),
                    }
                }
            }
        }
    }
}

/// Export logs to a downloadable file
fn export_logs(format: LogExportFormat, state: &Signal<LogsState>, toasts: Toasts) {
    let current_state = state.read();
    let storage = get_logs_storage();
    let exported_data = storage.export_logs(
        format,
        LogsQuery::new()
            .with_search(&current_state.search_query)
            .with_level_filters(Cow::Borrowed(&current_state.selected_levels))
            .with_limit(current_state.max_display),
    );

    let (file_name, content_type) = match format {
        LogExportFormat::Text => ("logs.txt", "text/plain;charset=utf-8"),
        LogExportFormat::Json => ("logs.json", "application/json"),
        LogExportFormat::Csv => ("logs.csv", "text/csv;charset=utf-8"),
    };

    spawn_forever(async move {
        let saved = save_text_file(TextFile {
            file_name,
            content_type,
            content: &exported_data,
        })
        .await;
        match saved {
            Ok(()) => {
                toasts.success(
                    format!("Exported {file_name}"),
                    "The log entries on screen were saved.",
                );
            }
            Err(error) => {
                tracing::error!("Failed to export the logs: {error}");
                toasts.error("Couldn't export the logs", error.to_string());
            }
        }
    });
}
