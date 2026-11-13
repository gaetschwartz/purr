//! The History screen: every saved transcription, searchable, with rename, export and delete.

mod clear_dialog;
pub mod delete_dialog;
mod export_host;
pub mod rename_dialog;
mod row;
mod unreadable_notice;

use crate::{
    components::{
        banner::{Banner, BannerTone},
        button::{Button, ButtonSize, ButtonVariant},
        empty_state::EmptyState,
        icon::IconName,
        page::Page,
        search_field::SearchField,
        spinner::Spinner,
    },
    history::{
        filter::matching,
        store::{HistoryLoad, use_history},
        wording::transcriptions,
    },
    main_app::Route,
    utils::clock::unix_ms,
};
use clear_dialog::ClearDialog;
use dioxus::prelude::*;
use row::HistoryRow;
use unreadable_notice::UnreadableNotice;

/// Saved transcriptions, newest first.
#[component]
pub fn History() -> Element {
    let store = use_history();
    let navigator = use_navigator();
    let mut query = use_signal(String::new);
    let mut clearing = use_signal(|| false);

    let listing = match store.load() {
        HistoryLoad::Loading => {
            return rsx! {
                Page { title: "History", wide: true,
                    div { class: "row", role: "status",
                        Spinner {}
                        span { class: "muted", "Opening your history…" }
                    }
                }
            };
        }
        HistoryLoad::Failed(message) => {
            return rsx! {
                Page { title: "History", wide: true,
                    Banner {
                        tone: BannerTone::Bad,
                        title: "Purr couldn't open your history",
                        role: "alert",
                        actions: rsx! {
                            Button {
                                variant: ButtonVariant::Primary,
                                size: ButtonSize::Small,
                                icon: IconName::Retry,
                                onclick: move |_| store.reload(),
                                "Try again"
                            }
                        },
                        "{message}"
                    }
                }
            };
        }
        HistoryLoad::Ready(listing) => listing,
    };

    let now = unix_ms();
    let total = listing.summaries.len();
    let shown = matching(&listing.summaries, &query());
    let searching = !query().trim().is_empty();

    if total == 0 {
        return rsx! {
            Page { title: "History", wide: true,
                UnreadableNotice { unreadable: listing.unreadable.clone() }
                EmptyState {
                    icon: IconName::Cat,
                    title: "No transcriptions yet",
                    actions: rsx! {
                        Button {
                            variant: ButtonVariant::Primary,
                            icon: IconName::Plus,
                            id: "empty-go",
                            onclick: move |_| {
                                navigator.push(Route::Home {});
                            },
                            "Transcribe a file"
                        }
                    },
                }
            }
        };
    }

    rsx! {
        Page {
            title: "History",
            subtitle: transcriptions(total),
            wide: true,
            actions: rsx! {
                div { class: "row wrap",
                    SearchField {
                        id: "h-search",
                        value: query(),
                        placeholder: "Search history",
                        label: "Search history",
                        described_by: "h-count",
                        style: "width:240px;max-width:100%",
                        oninput: move |value: String| query.set(value),
                    }
                    Button {
                        variant: ButtonVariant::Danger,
                        icon: IconName::Trash,
                        id: "history-clear",
                        aria_haspopup: "dialog",
                        onclick: move |_| clearing.set(true),
                        "Clear all"
                    }
                }
            },
            UnreadableNotice { unreadable: listing.unreadable.clone() }
            p { class: "sr", id: "h-count", role: "status", aria_live: "polite",
                if searching {
                    "{shown.len()} of {transcriptions(total)} match."
                }
            }
            if shown.is_empty() {
                EmptyState { icon: IconName::Search, title: "Nothing matches", inline: true,
                    "No saved transcription has “{query().trim()}” in its name or text."
                }
            } else {
                section {
                    class: "group surface",
                    id: "hlist",
                    aria_label: "Transcriptions",
                    for summary in shown {
                        HistoryRow { key: "{summary.id}", summary: summary.clone(), now_ms: now }
                    }
                }
            }
        }
        if clearing() {
            ClearDialog {
                count: total,
                unreadable: listing.unreadable.len(),
                on_close: move |()| clearing.set(false),
            }
        }
    }
}
