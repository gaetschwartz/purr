use super::marked_text::MarkedText;
use crate::{
    components::audio_player::PlayerHandle,
    transcribe::{row_text::word_spans, timing},
};
use dioxus::prelude::*;
use purr_common::platform::TranscriptSegment;
use std::ops::Range;

/// One row of the Segments layout.
///
/// With a `player` the row seeks: the whole row is one button, or, when `word_seek` is on and the
/// segment has word times, the timestamp is the button and the words take pointer clicks.
#[component]
pub fn SegmentRow(
    segment: TranscriptSegment,
    matches: (Vec<Range<usize>>, usize),
    current: Option<usize>,
    unsure: Vec<usize>,
    playing: bool,
    player: Option<PlayerHandle>,
    word_seek: bool,
    live: bool,
) -> Element {
    let start = segment.start_time;
    let clock = timing::clock(start);
    let words = word_spans(&segment.text, &segment.words);
    let by_word = word_seek && words.iter().any(|word| word.start.is_some());
    let line = |seek: Option<PlayerHandle>| {
        rsx! {
            span { class: if seek.is_some() { "tx words" } else { "tx" }, dir: "auto",
                MarkedText {
                    text: segment.text.trim().to_owned(),
                    words: words.clone(),
                    unsure: unsure.clone(),
                    matches: matches.clone(),
                    current,
                    seek,
                }
            }
        }
    };
    match player {
        Some(player) if by_word => rsx! {
            div { class: if playing { "seg-row split playing" } else { "seg-row split" },
                button {
                    class: "ts",
                    r#type: "button",
                    aria_current: playing.then_some("true"),
                    onclick: move |_| player.play_from(start),
                    span { class: "sr", "Play from " }
                    "{clock}"
                }
                {line(Some(player))}
            }
        },
        Some(player) => rsx! {
            button {
                class: if playing { "seg-row playing" } else { "seg-row" },
                r#type: "button",
                aria_current: playing.then_some("true"),
                onclick: move |_| player.play_from(start),
                span { class: "ts",
                    span { class: "sr", "Play from " }
                    "{clock}"
                }
                {line(None)}
            }
        },
        None => rsx! {
            div { class: if live { "seg-row still new" } else { "seg-row still" },
                span { class: "ts", "{clock}" }
                {line(None)}
            }
        },
    }
}

/// The row for the uncommitted end of a run.
#[component]
pub fn TentativeRow(segment: TranscriptSegment) -> Element {
    let clock = timing::clock(segment.start_time);
    rsx! {
        div { class: "seg-row still new tentative",
            span { class: "ts", "{clock}" }
            span { class: "tx", dir: "auto",
                "{segment.text.trim()}"
                span { class: "caret", aria_hidden: "true" }
            }
        }
    }
}
