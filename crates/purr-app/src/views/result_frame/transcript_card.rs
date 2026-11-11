use crate::{
    components::{
        audio_player::PlayerHandle,
        icon::IconName,
        icon_button::IconButton,
        search_field::SearchField,
        segmented::{Segment, Segmented},
        switch::Switch,
    },
    transcribe::{
        playback::segment_at,
        reading::{SEGMENTS_PER_PARAGRAPH, paragraphs},
        search::{self, Match, Piece, find_all, pieces},
        timing,
    },
};
use dioxus::prelude::*;
use purr_common::platform::TranscriptSegment;
use std::{ops::Range, rc::Rc};

/// The DOM id of the transcript search field.
pub const SEARCH_ID: &str = "t-search";

const REVEAL_CURRENT: &str =
    "document.querySelector('#segs mark.cur')?.scrollIntoView({block:'center'})";

/// How the transcript is laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    Segments,
    Reading,
}

/// The ranges of the matches in segment `segment`, and the number of the first one.
fn ranges_of(matches: &[Match], segment: usize) -> (Vec<Range<usize>>, usize) {
    let first = matches
        .iter()
        .position(|found| found.segment == segment)
        .unwrap_or(0);
    let ranges = matches
        .iter()
        .filter(|found| found.segment == segment)
        .map(|found| found.range.clone())
        .collect();
    (ranges, first)
}

/// The transcript with search, the Segments or Reading layout and the timestamps toggle.
#[component]
pub fn TranscriptCard(
    segments: Rc<Vec<TranscriptSegment>>,
    /// Absent when the audio is not available, which turns the lines into plain text.
    player: Option<PlayerHandle>,
) -> Element {
    let mut query = use_signal(String::new);
    let mut current = use_signal(|| 0_usize);
    let mut layout = use_signal(|| Layout::Segments);
    let mut show_times = use_signal(|| true);

    let texts: Vec<&str> = segments.iter().map(|segment| segment.text.trim()).collect();
    let matches = find_all(texts.iter().copied(), &query());
    let count = matches.len();
    let active = current().min(count.saturating_sub(1));
    let has_query = !query().trim().is_empty();
    let playing = player.and_then(|player| {
        segment_at(&segments, player.position())
            .filter(|_| player.is_playing() || player.position() > 0.0)
    });

    let mut step = move |forward: bool| {
        if count == 0 {
            return;
        }
        current.set(if forward {
            search::next(active, count)
        } else {
            search::previous(active, count)
        });
        document::eval(REVEAL_CURRENT);
    };

    let status = if !has_query {
        String::new()
    } else if count == 0 {
        "No matches".to_owned()
    } else {
        format!("{} of {count}", active + 1)
    };

    rsx! {
        section { class: "card glass tight", aria_label: "Transcript",
            div { class: "tools",
                SearchField {
                    id: SEARCH_ID,
                    value: query(),
                    placeholder: "Search transcript",
                    label: "Search transcript",
                    described_by: "t-count",
                    oninput: move |value: String| {
                        query.set(value);
                        current.set(0);
                    },
                    onkeydown: move |event: KeyboardEvent| {
                        if event.key() == Key::Named(NamedKey::Enter) {
                            event.prevent_default();
                            step(!event.modifiers().shift());
                        }
                    },
                }
                span { class: if has_query { "searchnav" } else { "sr" }, id: "t-count", aria_live: "polite",
                    "{status}"
                    if has_query {
                        IconButton {
                            icon: IconName::ChevronUp,
                            label: "Previous match",
                            disabled: count == 0,
                            onclick: move |_| step(false),
                        }
                        IconButton {
                            icon: IconName::ChevronDown,
                            label: "Next match",
                            disabled: count == 0,
                            onclick: move |_| step(true),
                        }
                    }
                }
                span { class: "spacer" }
                Segmented {
                    label: "Transcript layout",
                    value: layout(),
                    segments: vec![
                        Segment::new(Layout::Segments, "Segments").with_icon(IconName::List),
                        Segment::new(Layout::Reading, "Reading").with_icon(IconName::Text),
                    ],
                    onchange: move |next| layout.set(next),
                }
                span { class: "inline-sw",
                    span { id: "ts-label", "Timestamps" }
                    Switch {
                        labelled_by: "ts-label",
                        checked: show_times(),
                        onchange: move |on| show_times.set(on),
                    }
                }
            }
            div { style: "padding:10px",
                if segments.is_empty() {
                    p { class: "faint", style: "margin:0;padding:6px 10px", "No speech was recognised in this audio." }
                } else {
                    match layout() {
                        Layout::Segments => rsx! {
                            div { class: if show_times() { "live" } else { "live no-ts" }, id: "segs",
                                for (index , segment) in segments.iter().enumerate() {
                                    SegmentRow {
                                        key: "{index}",
                                        segment: segment.clone(),
                                        matches: ranges_of(&matches, index),
                                        current: has_query.then_some(active),
                                        playing: playing == Some(index),
                                        player,
                                    }
                                }
                            }
                        },
                        Layout::Reading => rsx! {
                            div { class: "para", id: "segs",
                                for range in paragraphs(segments.len(), SEGMENTS_PER_PARAGRAPH) {
                                    p { key: "{range.start}",
                                        if show_times() {
                                            span { class: "faint tnum", style: "font-size:13px;margin-right:8px",
                                                {segments.get(range.start).map(|segment| timing::clock(segment.start_time))}
                                            }
                                        }
                                        for index in range {
                                            if let Some(segment) = segments.get(index) {
                                                Marked {
                                                    key: "{index}",
                                                    text: segment.text.trim().to_owned(),
                                                    matches: ranges_of(&matches, index),
                                                    current: has_query.then_some(active),
                                                }
                                                " "
                                            }
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            }
        }
    }
}

#[component]
fn SegmentRow(
    segment: TranscriptSegment,
    matches: (Vec<Range<usize>>, usize),
    current: Option<usize>,
    playing: bool,
    player: Option<PlayerHandle>,
) -> Element {
    let start = segment.start_time;
    let clock = timing::clock(start);
    let line = rsx! {
        span { class: "tx",
            Marked {
                text: segment.text.trim().to_owned(),
                matches,
                current,
            }
        }
    };
    match player {
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
                {line}
            }
        },
        None => rsx! {
            div { class: "seg-row still",
                span { class: "ts", "{clock}" }
                {line}
            }
        },
    }
}

#[component]
fn Marked(text: String, matches: (Vec<Range<usize>>, usize), current: Option<usize>) -> Element {
    let (ranges, first) = matches;
    rsx! {
        for (position , piece) in pieces(&text, &ranges, first).into_iter().enumerate() {
            match piece {
                Piece::Plain(plain) => rsx! {
                    span { key: "{position}", "{plain}" }
                },
                Piece::Marked { text, index } => rsx! {
                    mark { key: "{position}", class: if current == Some(index) { "cur" }, "{text}" }
                },
            }
        }
    }
}
