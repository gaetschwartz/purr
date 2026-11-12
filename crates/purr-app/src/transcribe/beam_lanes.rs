//! The decoder's beams written as lanes of text for the live view.

use super::timing;
use purr_common::beams::{Beam, BeamSnapshot};
use std::time::Duration;

/// Tokens the decoder gave less than this probability are underlined.
pub const UNSURE_BELOW: f32 = 0.5;

const REFRESH: Duration = Duration::from_millis(100);
const REFRESH_REDUCED: Duration = Duration::from_millis(500);

/// How often the lanes redraw.
#[must_use]
pub const fn refresh_interval(reduced_motion: bool) -> Duration {
    if reduced_motion {
        REFRESH_REDUCED
    } else {
        REFRESH
    }
}

/// A piece of a lane's text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneToken<'a> {
    pub text: &'a str,
    pub unsure: bool,
}

/// One hypothesis as a line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lane<'a> {
    /// From 1, best first
    pub rank: usize,
    pub leader: bool,
    pub tokens: Vec<LaneToken<'a>>,
    /// The sum of the log probabilities, with a real minus sign
    pub score: String,
}

fn lane(index: usize, beam: &Beam) -> Lane<'_> {
    Lane {
        rank: index + 1,
        leader: index == 0,
        tokens: beam
            .tokens
            .iter()
            .map(|token| LaneToken {
                text: &token.text,
                unsure: token.p < UNSURE_BELOW,
            })
            .collect(),
        score: format!("{:.2}", beam.log_prob).replace('-', "\u{2212}"),
    }
}

/// The lanes of `snapshot`, best first.
#[must_use]
pub fn lanes(snapshot: &BeamSnapshot) -> Vec<Lane<'_>> {
    snapshot
        .beams
        .iter()
        .enumerate()
        .map(|(index, beam)| lane(index, beam))
        .collect()
}

/// The window being decoded and how far the decoder is in it: `0:29–0:59 · step 12`.
#[must_use]
pub fn header(snapshot: &BeamSnapshot) -> String {
    format!(
        "{}\u{2013}{} \u{b7} step {}",
        timing::clock(snapshot.window.0),
        timing::clock(snapshot.window.1),
        snapshot.step
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::beams::BeamToken;

    fn token(text: &str, p: f32) -> BeamToken {
        BeamToken {
            text: text.to_owned(),
            p,
        }
    }

    fn snapshot() -> BeamSnapshot {
        BeamSnapshot {
            window: (89.0, 119.5),
            attempt: 0,
            step: 12,
            beams: vec![
                Beam {
                    tokens: vec![token(" we", 0.96), token(" Thursday", 0.71)],
                    log_prob: -2.614,
                },
                Beam {
                    tokens: vec![token(" we", 0.96), token(" Tuesday", 0.22)],
                    log_prob: -3.0,
                },
            ],
        }
    }

    #[test]
    fn lanes_rank_the_beams_and_flag_unsure_tokens() {
        let snapshot = snapshot();
        assert_eq!(
            lanes(&snapshot),
            vec![
                Lane {
                    rank: 1,
                    leader: true,
                    tokens: vec![
                        LaneToken {
                            text: " we",
                            unsure: false
                        },
                        LaneToken {
                            text: " Thursday",
                            unsure: false
                        },
                    ],
                    score: "\u{2212}2.61".to_owned(),
                },
                Lane {
                    rank: 2,
                    leader: false,
                    tokens: vec![
                        LaneToken {
                            text: " we",
                            unsure: false
                        },
                        LaneToken {
                            text: " Tuesday",
                            unsure: true
                        },
                    ],
                    score: "\u{2212}3.00".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn a_token_at_the_threshold_is_not_flagged() {
        let beam = Beam {
            tokens: vec![token("a", UNSURE_BELOW), token("b", 0.49)],
            log_prob: 0.0,
        };
        let lane = lane(0, &beam);
        assert_eq!(
            lane.tokens
                .iter()
                .map(|token| token.unsure)
                .collect::<Vec<_>>(),
            vec![false, true]
        );
        assert_eq!(lane.score, "0.00");
    }

    #[test]
    fn the_header_names_the_window_and_the_step() {
        assert_eq!(header(&snapshot()), "1:29\u{2013}1:59 \u{b7} step 12");
    }

    #[test]
    fn reduced_motion_redraws_five_times_slower() {
        assert_eq!(refresh_interval(false), Duration::from_millis(100));
        assert_eq!(refresh_interval(true), Duration::from_millis(500));
    }
}
