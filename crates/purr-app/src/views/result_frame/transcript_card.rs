use super::{
    beam_panel::{BeamsView, InlineBeams},
    marked_text::MarkedText,
    more_menu::{MORE_ID, MoreAction, MoreMenu, TranscriptionFacts},
    segment_row::{SegmentRow, TentativeRow},
};
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
    platform::IS_WEB,
    settings_store::use_settings,
    transcribe::{
        live_rows::{Row, rows},
        playback::segment_at,
        reading::{SEGMENTS_PER_PARAGRAPH, paragraphs},
        row_text::{needed_words, word_spans},
        search::{self, Match, find_all},
        timing,
        unsure::unsure_words,
        view_options::{Evidence, Moment, details},
    },
};
use dioxus::prelude::*;
use purr_common::{
    export_format::ExportFormat, platform::TranscriptSegment, transcript_view::TranscriptLayout,
};
use std::{ops::Range, rc::Rc};

/// The DOM id of the transcript search field.
pub const SEARCH_ID: &str = "t-search";

const TENTATIVE_KEY: &str = "tentative";

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

fn evidence(segments: &[TranscriptSegment], playable: bool) -> Evidence {
    let words = || segments.iter().flat_map(|segment| segment.words.iter());
    Evidence {
        scores: words().any(|word| word.score.is_some()),
        times: words().any(|word| word.start_cs.is_some()),
        playable,
    }
}

/// The transcript with its toolbar: search, Copy, Export and the More menu.
#[component]
pub fn TranscriptCard(
    segments: Rc<Vec<TranscriptSegment>>,
    /// The uncommitted end of a run, shown after the segments while `live`.
    tentative: Option<TranscriptSegment>,
    /// Absent while the run is going or when the audio is not available: the lines are plain text.
    player: Option<PlayerHandle>,
    /// The audio can be played, though not necessarily yet.
    playable: bool,
    /// The run is still writing the transcript.
    #[props(default)]
    live: bool,
    moment: Moment,
    beams: BeamsView,
    export: ExportSource,
    /// Export waits in its place until the transcript is final.
    #[props(default = true)]
    export_shown: bool,
    facts: TranscriptionFacts,
    more: Vec<MoreAction>,
    on_more: EventHandler<MoreAction>,
) -> Element {
    let toasts = use_toasts();
    let settings = use_settings();
    let view = settings.with(|settings| settings.ui.transcript);
    let mut query = use_signal(String::new);
    let mut current = use_signal(|| 0_usize);
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
    let unsure = if view.mark_unsure {
        unsure_words(&segments, view.unsure_share)
    } else {
        Vec::new()
    };
    let unsure_of = |index: usize| unsure.get(index).cloned().unwrap_or_default();

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
    let list_class = if view.show_times {
        "live"
    } else {
        "live no-ts"
    };
    let shown_tentative = tentative.as_ref().filter(|_| live);

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
            if !IS_WEB {
                InlineBeams { beams }
            }
            div { class: "tbody",
                if segments.is_empty() && !live {
                    p { class: "faint", style: "margin:0;padding:6px 10px", "No speech was recognised in this audio." }
                } else {
                    match view.layout {
                        TranscriptLayout::Segments => rsx! {
                            div { class: list_class, id: "segs",
                                for row in rows(&segments, shown_tentative, live) {
                                    match row {
                                        Row::Committed(index, segment) => rsx! {
                                            SegmentRow {
                                                key: "{index}",
                                                segment: segment.clone(),
                                                matches: ranges_of(&matches, index),
                                                current: has_query.then_some(active),
                                                unsure: unsure_of(index),
                                                playing: playing == Some(index),
                                                player,
                                                word_seek: view.word_seek,
                                                live,
                                            }
                                        },
                                        Row::Tentative(segment) => rsx! {
                                            TentativeRow { key: "{TENTATIVE_KEY}", segment: segment.clone() }
                                        },
                                    }
                                }
                            }
                        },
                        TranscriptLayout::Reading => rsx! {
                            div { class: "para", id: "segs",
                                for range in paragraphs(segments.len(), SEGMENTS_PER_PARAGRAPH) {
                                    p { key: "{range.start}", dir: "auto",
                                        if view.show_times {
                                            span { class: "faint tnum", style: "font-size:13px;margin-right:8px",
                                                {segments.get(range.start).map(|segment| timing::clock(segment.start_time))}
                                            }
                                        }
                                        for index in range {
                                            if let Some(segment) = segments.get(index) {
                                                MarkedText {
                                                    key: "{index}",
                                                    text: segment.text.trim().to_owned(),
                                                    words: needed_words(
                                                        word_spans(&segment.text, &segment.words),
                                                        &unsure_of(index),
                                                        false,
                                                    ),
                                                    unsure: unsure_of(index),
                                                    matches: ranges_of(&matches, index),
                                                    current: has_query.then_some(active),
                                                    seek: None,
                                                }
                                                " "
                                            }
                                        }
                                    }
                                }
                                if let Some(segment) = shown_tentative {
                                    p { key: "{TENTATIVE_KEY}", class: "tentative", dir: "auto",
                                        "{segment.text.trim()}"
                                        span { class: "caret", aria_hidden: "true" }
                                    }
                                }
                            }
                        },
                    }
                }
                if live && shown_tentative.is_none() {
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
                details: details(IS_WEB, moment, evidence(&segments, playable)),
                on_close: move |()| more_open.set(false),
            }
        }
    }
}
