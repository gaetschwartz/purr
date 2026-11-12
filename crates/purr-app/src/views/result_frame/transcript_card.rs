use super::more_menu::{Layout, MORE_ID, MoreAction, MoreMenu, TranscriptionFacts};
use crate::{
    components::{
        audio_player::PlayerHandle,
        button::{Button, ButtonVariant},
        export_menu::{ExportMenu, ExportSource, copy_export},
        icon::IconName,
        icon_button::IconButton,
        search_field::SearchField,
        toast::use_toasts,
    },
    transcribe::{
        playback::segment_at,
        reading::{SEGMENTS_PER_PARAGRAPH, paragraphs},
        search::{self, Match, Piece, find_all, pieces},
        timing,
    },
};
use dioxus::prelude::*;
use purr_common::{export_format::ExportFormat, platform::TranscriptSegment};
use std::{ops::Range, rc::Rc};

/// The DOM id of the transcript search field.
pub const SEARCH_ID: &str = "t-search";

const REVEAL_CURRENT: &str =
    "document.querySelector('#segs mark.cur')?.scrollIntoView({block:'center'})";

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

/// The transcript with its toolbar: search, Copy, Export and the More menu.
#[component]
pub fn TranscriptCard(
    segments: Rc<Vec<TranscriptSegment>>,
    /// Absent while the run is going or when the audio is not available: the lines are plain text.
    player: Option<PlayerHandle>,
    /// The run is still writing the transcript.
    #[props(default)]
    live: bool,
    export: ExportSource,
    /// Export waits in its place until the transcript is final.
    #[props(default = true)]
    export_shown: bool,
    facts: TranscriptionFacts,
    more: Vec<MoreAction>,
    on_more: EventHandler<MoreAction>,
) -> Element {
    let toasts = use_toasts();
    let mut query = use_signal(String::new);
    let mut current = use_signal(|| 0_usize);
    let layout = use_signal(|| Layout::Segments);
    let show_times = use_signal(|| true);
    let mut more_open = use_signal(|| false);

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
    let copy_source = export.clone();
    let list_class = if show_times() { "live" } else { "live no-ts" };

    rsx! {
        section { class: "card glass tight tcard", aria_label: "Transcript",
            div { class: "tools",
                SearchField {
                    id: SEARCH_ID,
                    value: query(),
                    placeholder: "Search",
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
                span { class: "tacts",
                    Button {
                        variant: ButtonVariant::Tonal,
                        icon: IconName::Copy,
                        id: "copy-btn",
                        disabled: segments.is_empty(),
                        onclick: move |_| copy_export(&copy_source, ExportFormat::Txt, toasts),
                        "Copy"
                    }
                    span {
                        class: "export-slot",
                        "data-off": (!export_shown).then_some("true"),
                        ExportMenu {
                            id: "export-btn",
                            source: export,
                            available: export_shown,
                        }
                    }
                }
                IconButton {
                    icon: IconName::More,
                    label: "More",
                    id: MORE_ID,
                    aria_haspopup: "dialog",
                    aria_expanded: more_open(),
                    onclick: move |_| more_open.toggle(),
                }
            }
            div { class: "tbody",
                if segments.is_empty() && !live {
                    p { class: "faint", style: "margin:0;padding:6px 10px", "No speech was recognised in this audio." }
                } else {
                    match layout() {
                        Layout::Segments => rsx! {
                            div { class: list_class, id: "segs",
                                for (index , segment) in segments.iter().enumerate() {
                                    SegmentRow {
                                        key: "{index}",
                                        segment: segment.clone(),
                                        matches: ranges_of(&matches, index),
                                        current: has_query.then_some(active),
                                        playing: playing == Some(index),
                                        player,
                                        live,
                                    }
                                }
                            }
                        },
                        Layout::Reading => rsx! {
                            div { class: "para", id: "segs",
                                for range in paragraphs(segments.len(), SEGMENTS_PER_PARAGRAPH) {
                                    p { key: "{range.start}", dir: "auto",
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
                if live {
                    div { class: "seg-row caret-row", aria_hidden: "true",
                        span { class: "ts" }
                        span { class: "tx",
                            span { class: "caret" }
                        }
                    }
                }
            }
        }
        if more_open() {
            MoreMenu {
                facts,
                actions: more,
                on_action: on_more,
                layout,
                show_times,
                on_close: move |()| more_open.set(false),
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
    live: bool,
) -> Element {
    let start = segment.start_time;
    let clock = timing::clock(start);
    let line = rsx! {
        span { class: "tx", dir: "auto",
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
            div { class: if live { "seg-row still new" } else { "seg-row still" },
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
