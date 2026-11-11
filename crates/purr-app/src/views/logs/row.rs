use crate::platform::logging::{LogEntry, SerdeTracingLevel};
use dioxus::prelude::*;
use std::sync::Arc;

const fn badge_class(level: SerdeTracingLevel) -> &'static str {
    match level {
        SerdeTracingLevel::ERROR => "lv error",
        SerdeTracingLevel::WARN => "lv warn",
        SerdeTracingLevel::INFO => "lv info",
        SerdeTracingLevel::DEBUG => "lv debug",
        SerdeTracingLevel::TRACE => "lv trace",
    }
}

/// The level named in full in the badge, so it never relies on colour alone.
#[must_use]
pub const fn level_label(level: SerdeTracingLevel) -> &'static str {
    match level {
        SerdeTracingLevel::ERROR => "Error",
        SerdeTracingLevel::WARN => "Warning",
        SerdeTracingLevel::INFO => "Info",
        SerdeTracingLevel::DEBUG => "Debug",
        SerdeTracingLevel::TRACE => "Trace",
    }
}

fn origin(entry: &LogEntry) -> String {
    match (&entry.file, entry.line) {
        (Some(file), Some(line)) => format!("{} ({file}:{line})", entry.target),
        (Some(file), None) => format!("{} ({file})", entry.target),
        (None, _) => entry.target.clone(),
    }
}

/// One entry of the log: time, level, target and message, with its fields after the message.
#[component]
pub fn LogRow(entry: Arc<LogEntry>) -> Element {
    let class = if entry.id.is_multiple_of(2) {
        "log alt"
    } else {
        "log"
    };
    rsx! {
        div { class,
            span { class: "t", {entry.formatted_timestamp()} }
            span { class: badge_class(entry.level), {level_label(entry.level)} }
            span { class: "tg", title: origin(&entry), {entry.short_target()} }
            span { class: "m",
                "{entry.message}"
                for (key , value) in entry.fields.iter() {
                    span { class: "kv", " {key}={value}" }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(file: Option<&str>, line: Option<u32>) -> LogEntry {
        LogEntry {
            id: 1,
            timestamp: 0,
            level: SerdeTracingLevel::INFO,
            target: "purr_app::views::logs".to_owned(),
            message: String::new(),
            fields: vec![],
            file: file.map(str::to_owned),
            line,
        }
    }

    #[test]
    fn the_origin_names_the_target_and_where_it_was_logged() {
        assert_eq!(
            origin(&entry(Some("logs.rs"), Some(12))),
            "purr_app::views::logs (logs.rs:12)"
        );
        assert_eq!(
            origin(&entry(Some("logs.rs"), None)),
            "purr_app::views::logs (logs.rs)"
        );
        assert_eq!(origin(&entry(None, Some(12))), "purr_app::views::logs");
    }

    #[test]
    fn warnings_are_spelled_out_for_the_badge() {
        assert_eq!(level_label(SerdeTracingLevel::WARN), "Warning");
        assert_eq!(level_label(SerdeTracingLevel::ERROR), "Error");
    }
}
