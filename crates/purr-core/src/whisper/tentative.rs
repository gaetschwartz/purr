//! When a stream decodes its uncommitted audio only to show it early.

use crate::config::TentativeText;

/// Audio held back from decoding while the stream waits for its source, in seconds of audio
const MIN_PENDING_SECONDS: f64 = 1.0;
/// What a decode is taken to cost until one has been timed, in wall seconds
const ASSUMED_DECODE_SECONDS: f64 = 1.0;

/// The state of a stream that has no window to decode and waits for its source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Waiting {
    /// Seconds of audio received since the last tentative decode, or since the last commit
    pub new_audio: f64,
    /// Seconds of audio not committed yet
    pub pending: f64,
    /// Seconds of audio still to receive before the next window is ready to decode
    pub until_window: f64,
    /// Seconds of audio the source delivers per wall second, once it has delivered any
    pub arrival_rate: Option<f64>,
    /// Wall seconds the last decode took
    pub last_decode: Option<f64>,
}

/// Whether to decode the pending audio tentatively now.
///
/// Only while waiting, and only when the source is slow enough that the next window cannot
/// be ready before the decode ends, so a tentative decode never holds back committed text.
/// A source that delivers faster than it can be decoded, such as a file, never waits that long.
#[must_use]
pub fn due(policy: TentativeText, waiting: Waiting) -> bool {
    match policy {
        TentativeText::Off | TentativeText::Decoded => false,
        TentativeText::WhileWaiting { every_seconds } => {
            let Some(rate) = waiting.arrival_rate.filter(|rate| *rate > 0.0) else {
                return false;
            };
            let decode = waiting.last_decode.unwrap_or(ASSUMED_DECODE_SECONDS);
            waiting.new_audio >= f64::from(every_seconds)
                && waiting.pending >= MIN_PENDING_SECONDS
                && waiting.until_window / rate > decode
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EVERY_3S: TentativeText = TentativeText::WhileWaiting { every_seconds: 3.0 };

    fn live() -> Waiting {
        Waiting {
            new_audio: 3.0,
            pending: 6.0,
            until_window: 24.0,
            arrival_rate: Some(1.0),
            last_decode: Some(1.5),
        }
    }

    #[test]
    fn a_live_source_with_enough_new_audio_is_decoded_tentatively() {
        assert!(due(EVERY_3S, live()));
    }

    #[test]
    fn only_the_waiting_policy_decodes_tentatively() {
        assert!(!due(TentativeText::Off, live()));
        assert!(!due(TentativeText::Decoded, live()));
    }

    #[test]
    fn too_little_new_or_pending_audio_waits() {
        assert!(!due(
            EVERY_3S,
            Waiting {
                new_audio: 2.9,
                ..live()
            }
        ));
        assert!(!due(
            EVERY_3S,
            Waiting {
                new_audio: 3.0,
                pending: 0.5,
                ..live()
            }
        ));
    }

    #[test]
    fn a_window_due_before_the_decode_ends_is_not_held_back() {
        assert!(!due(
            EVERY_3S,
            Waiting {
                until_window: 1.0,
                ..live()
            }
        ));
        assert!(!due(
            EVERY_3S,
            Waiting {
                arrival_rate: Some(40.0),
                ..live()
            }
        ));
    }

    #[test]
    fn an_untimed_decode_is_taken_to_cost_a_second() {
        let untimed = Waiting {
            last_decode: None,
            ..live()
        };
        assert!(due(EVERY_3S, untimed));
        assert!(!due(
            EVERY_3S,
            Waiting {
                until_window: 0.9,
                ..untimed
            }
        ));
    }

    #[test]
    fn a_source_that_delivered_nothing_is_not_decoded() {
        assert!(!due(
            EVERY_3S,
            Waiting {
                arrival_rate: None,
                ..live()
            }
        ));
    }
}
