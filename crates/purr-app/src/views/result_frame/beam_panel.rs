use crate::transcribe::beam_lanes::{header, lanes};
use dioxus::prelude::*;
use purr_common::beams::BeamSnapshot;

/// What the beam view shows: whether it is wanted now, and the latest hypotheses.
///
/// The snapshot outlives `on` so the panel keeps its lanes while it folds away.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BeamsView {
    pub on: bool,
    pub snapshot: Option<BeamSnapshot>,
}

/// The lanes of the decoder's beams, best first.
#[component]
fn BeamLanes(snapshot: Option<BeamSnapshot>) -> Element {
    let Some(snapshot) = snapshot else {
        return rsx! {
            p { class: "faint", style: "margin:0;padding:4px 6px;font-size:12.5px", "Waiting for the decoder…" }
        };
    };
    rsx! {
        ol {
            for lane in lanes(&snapshot) {
                li { key: "{lane.rank}", class: if lane.leader { "lead" },
                    span { class: "rk", "{lane.rank}" }
                    span { class: "bt",
                        bdi {
                            "…"
                            for (position , token) in lane.tokens.iter().enumerate() {
                                if token.unsure {
                                    {
                                        let word = token.text.trim_start();
                                        let lead = token.text.get(..token.text.len() - word.len()).unwrap_or_default();
                                        rsx! {
                                            span { key: "{position}",
                                                "{lead}"
                                                span { class: "unsure", "{word}" }
                                            }
                                        }
                                    }
                                } else {
                                    span { key: "{position}", "{token.text}" }
                                }
                            }
                        }
                    }
                    span { class: "sc", "{lane.score}" }
                }
            }
        }
    }
}

fn window_text(snapshot: Option<&BeamSnapshot>) -> String {
    snapshot.map(header).unwrap_or_default()
}

/// The beams in a side card beside the transcript, when the app is wide enough.
#[component]
pub fn SideBeams(beams: BeamsView) -> Element {
    let window = window_text(beams.snapshot.as_ref());
    rsx! {
        aside { class: "side-beams", aria_label: "Decoder beams",
            div { class: "card surface",
                div { class: "beams",
                    div { class: "bh",
                        span { "Beams" }
                        span { "{window}" }
                    }
                    BeamLanes { snapshot: beams.snapshot }
                }
            }
        }
    }
}

/// The beams at the top of the transcript card, where the side card does not fit.
#[component]
pub fn InlineBeams(beams: BeamsView) -> Element {
    let window = window_text(beams.snapshot.as_ref());
    rsx! {
        div { class: "fold inline-beams", "data-off": (!beams.on).then_some("true"),
            div {
                details { class: "beams", open: true,
                    summary {
                        span { "Beams" }
                        span { "{window}" }
                    }
                    BeamLanes { snapshot: beams.snapshot }
                }
            }
        }
    }
}
