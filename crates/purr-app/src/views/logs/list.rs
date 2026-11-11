use super::{filter::Entries, row::LogRow};
use crate::components::{
    button::{Button, ButtonSize, ButtonVariant},
    empty_state::EmptyState,
};
use dioxus::prelude::*;
use std::{ops::Range, sync::Arc};

/// The scrollable region of rows, with a bar for entries left out above and the empty states.
///
/// `entries` are the matching entries and `rows` the positions shown; `recorded` says whether
/// anything was captured at all, which tells "nothing yet" from "nothing matches".
#[component]
pub fn LogList(
    entries: Entries,
    rows: Range<usize>,
    recorded: bool,
    on_earlier: EventHandler<()>,
    on_reset_filter: EventHandler<()>,
) -> Element {
    let hidden = rows.start;
    let visible = entries.get(rows).unwrap_or_default();
    rsx! {
        if hidden > 0 {
            div { class: "logmore",
                span { "{hidden} earlier entries are not shown." }
                Button {
                    variant: ButtonVariant::Plain,
                    size: ButtonSize::Small,
                    onclick: move |_| on_earlier.call(()),
                    "Show earlier"
                }
            }
        }
        div {
            class: "loglist",
            id: "loglist",
            role: "log",
            aria_label: "Log",
            aria_live: "off",
            tabindex: "0",
            if visible.is_empty() {
                if recorded {
                    EmptyState {
                        inline: true,
                        title: "No entries match",
                        actions: rsx! {
                            Button { onclick: move |_| on_reset_filter.call(()), "Show all entries" }
                        },
                        "Try a lower level or a different search."
                    }
                } else {
                    EmptyState {
                        inline: true,
                        title: "No log entries",
                        "New entries appear here as Purr works."
                    }
                }
            } else {
                for entry in visible {
                    LogRow { key: "{entry.id}", entry: Arc::clone(entry) }
                }
            }
        }
    }
}
