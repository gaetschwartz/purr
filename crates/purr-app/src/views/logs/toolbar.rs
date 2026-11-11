use super::{filter::LevelFilter, follow::FollowMode};
use crate::{
    components::{
        search_field::SearchField,
        segmented::{Segment, Segmented},
        switch::Switch,
    },
    utils::delay::sleep,
};
use dioxus::{core::Task, prelude::*};
use std::time::Duration;

/// How long the search waits for the next keystroke before filtering.
const SEARCH_DEBOUNCE: Duration = Duration::from_millis(200);

/// The level threshold, the search box and the auto-scroll switch.
///
/// `query` is the search text in force; typing updates the box at once and `on_query` after a pause.
#[component]
pub fn LogToolbar(
    level: LevelFilter,
    query: String,
    follow: FollowMode,
    on_level: EventHandler<LevelFilter>,
    on_query: EventHandler<String>,
    on_follow: EventHandler<bool>,
) -> Element {
    let mut draft = use_signal(|| query.clone());
    let mut pending = use_signal(|| None::<Task>);

    use_effect(use_reactive!(|query| {
        if *draft.peek() != query {
            draft.set(query);
        }
    }));

    let segments = LevelFilter::CHOICES
        .iter()
        .map(|(value, label)| Segment::new(*value, *label))
        .collect::<Vec<_>>();

    let oninput = move |text: String| {
        draft.set(text.clone());
        if let Some(task) = pending.write().take() {
            task.cancel();
        }
        let task = spawn(async move {
            sleep(SEARCH_DEBOUNCE).await;
            pending.set(None);
            on_query.call(text);
        });
        pending.set(Some(task));
    };

    rsx! {
        div { class: "logbar",
            Segmented {
                segments,
                value: level,
                onchange: move |value| on_level.call(value),
                label: "Minimum level",
            }
            SearchField {
                value: draft(),
                oninput,
                placeholder: "Search logs",
                label: "Search logs",
                id: "log-search",
            }
            span { class: "inline-sw",
                span { id: "log-follow-label", "Auto-scroll" }
                Switch {
                    checked: follow.is_following(),
                    labelled_by: "log-follow-label",
                    onchange: move |on| on_follow.call(on),
                }
            }
        }
    }
}
