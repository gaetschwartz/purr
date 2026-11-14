//! Where to cut a window again when the engine could not decode all of it.

use crate::engine::hypothesis::Span;

use super::{commit::WindowKind, pause::last_pause};

/// The shorter window to decode in place of one the engine stopped early on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shorter {
    /// In samples from the window start
    pub len: usize,
    pub kind: WindowKind,
}

/// Half of a window of `len` samples, ending in the last pause of `speech` when there is one.
///
/// `None` when half the window is shorter than `floor`: the text the engine gave is then all
/// there will be, and the window is committed with it.
#[must_use]
pub fn shorter(
    speech: Option<&[Span]>,
    len: usize,
    floor: usize,
    min_tail: usize,
) -> Option<Shorter> {
    let half = len / 2;
    if half < floor {
        return None;
    }
    let paused = speech.and_then(|speech| {
        let within: Vec<Span> = speech
            .iter()
            .filter(|span| span.start < half)
            .map(|span| Span {
                start: span.start,
                end: span.end.min(half),
            })
            .collect();
        last_pause(&within, half, floor, min_tail)
    });
    Some(match paused {
        Some(cut) => Shorter {
            len: cut,
            kind: WindowKind::Paused,
        },
        None => Shorter {
            len: half,
            kind: WindowKind::Full,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: usize = 16_000;

    fn span(start: f32, end: f32) -> Span {
        Span {
            start: (start * S as f32) as usize,
            end: (end * S as f32) as usize,
        }
    }

    #[test]
    fn without_a_speech_detector_the_window_is_halved() {
        assert_eq!(
            shorter(None, 30 * S, 5 * S, S),
            Some(Shorter {
                len: 15 * S,
                kind: WindowKind::Full
            })
        );
    }

    #[test]
    fn with_a_detector_the_cut_falls_in_the_last_pause_of_the_first_half() {
        let speech = [span(0.5, 6.0), span(8.0, 14.0), span(14.5, 25.0)];
        assert_eq!(
            shorter(Some(&speech), 30 * S, 5 * S, S),
            Some(Shorter {
                len: 14 * S + S / 4,
                kind: WindowKind::Paused
            })
        );
    }

    #[test]
    fn speech_running_to_the_middle_of_the_window_is_not_cut_there() {
        let speech = [span(0.5, 4.0), span(4.2, 25.0)];
        assert_eq!(
            shorter(Some(&speech), 30 * S, 5 * S, S),
            Some(Shorter {
                len: 15 * S,
                kind: WindowKind::Full
            })
        );
    }

    #[test]
    fn a_window_whose_half_is_under_the_floor_is_not_cut_again() {
        assert_eq!(shorter(None, 9 * S, 5 * S, S), None);
        assert_eq!(
            shorter(None, 10 * S, 5 * S, S),
            Some(Shorter {
                len: 5 * S,
                kind: WindowKind::Full
            })
        );
    }
}
