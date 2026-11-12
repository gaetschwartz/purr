//! The `--trace-beams` lines.

use crate::progress::clock;
use purr_common::beams::BeamSnapshot;
use std::fmt::Write as _;

/// One line for a decoding step: the window, the step and attempt, then each beam's log
/// probability and newest text, best first.
#[must_use]
pub fn line(snapshot: &BeamSnapshot) -> String {
    let mut line = format!(
        "[{}-{}] step {} try {}",
        clock(snapshot.window.0),
        clock(snapshot.window.1),
        snapshot.step,
        snapshot.attempt
    );
    for beam in &snapshot.beams {
        let text: String = beam
            .tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect();
        if write!(line, " | {:.2} {}", beam.log_prob, text.trim()).is_err() {
            break;
        }
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::beams::{Beam, BeamToken};

    fn token(text: &str) -> BeamToken {
        BeamToken {
            text: text.to_owned(),
            p: 0.5,
        }
    }

    #[test]
    fn a_step_reads_as_one_line_best_beam_first() {
        let snapshot = BeamSnapshot {
            window: (0.0, 11.0),
            attempt: 0,
            step: 6,
            beams: vec![
                Beam {
                    tokens: vec![token(" And"), token(" so"), token(" my")],
                    log_prob: -1.055,
                },
                Beam {
                    tokens: vec![token(" And"), token(" so"), token(",")],
                    log_prob: -2.084,
                },
            ],
        };
        assert_eq!(
            line(&snapshot),
            "[0:00-0:11] step 6 try 0 | -1.05 And so my | -2.08 And so,"
        );
    }
}
