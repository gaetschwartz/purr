//! Which windows of its source a transcription decodes.

use purr_common::model::profile::InputLimit;

use crate::{config::Windowing, engine::decoder::WindowLimits};

/// Audio arrives at this rate, in samples per second.
pub const SAMPLE_RATE: usize = 16_000;
const COMMIT_MARGIN_SECONDS: f32 = 1.0;
const MIN_WINDOW_SECONDS: f32 = 0.1;
const MIN_PAUSED_WINDOW_SECONDS: f32 = 1.0;
const MIN_RETRY_WINDOW_SECONDS: f32 = 5.0;

/// How the source is cut into windows; lengths in samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowPolicy {
    /// Windows of `max` samples, decoded once full or once the source ends: without a speech
    /// detector, segment ends in a short window are too loose to commit on
    Full { max: usize },
    /// Windows that grow by `step` new samples and end at a pause the speech detector finds,
    /// up to `max`
    AtPauses { step: usize, max: usize },
    /// The whole source in one decode, once it has all arrived
    Whole,
}

impl WindowPolicy {
    /// The policy for a decoder with `limits`, given whether a speech detector finds pauses
    /// and how many new samples a growing window waits for.
    ///
    /// The whole source goes in one decode only when the decoder takes audio of any length.
    #[must_use]
    pub fn choose(windowing: Windowing, limits: WindowLimits, detector: bool, step: usize) -> Self {
        let max = match limits.input {
            InputLimit::Unbounded => samples(limits.window.as_secs_f32()),
            InputLimit::UpTo(limit) => samples(limits.window.min(limit).as_secs_f32()),
        };
        match (windowing, limits.input) {
            (Windowing::Whole, InputLimit::Unbounded) => Self::Whole,
            (Windowing::Stream | Windowing::Whole, InputLimit::Unbounded | InputLimit::UpTo(_)) => {
                if detector {
                    Self::AtPauses { step, max }
                } else {
                    Self::Full { max }
                }
            }
        }
    }

    /// Whether the policy cuts windows at pauses, which needs a speech detector.
    #[must_use]
    pub const fn uses_pauses(self) -> bool {
        match self {
            Self::AtPauses { .. } => true,
            Self::Full { .. } | Self::Whole => false,
        }
    }
}

/// Sample counts every policy works with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Margins {
    /// A window shorter than this is not decoded
    pub min: usize,
    /// A window cut at a pause is at least this long
    pub min_paused: usize,
    /// Segments ending this close to a growing window's end are not committed
    pub commit: usize,
    /// A window the engine stopped early on is cut in half again only while the half is at
    /// least this long
    pub retry_floor: usize,
}

impl Default for Margins {
    fn default() -> Self {
        Self {
            min: samples(MIN_WINDOW_SECONDS),
            min_paused: samples(MIN_PAUSED_WINDOW_SECONDS),
            commit: samples(COMMIT_MARGIN_SECONDS),
            retry_floor: samples(MIN_RETRY_WINDOW_SECONDS),
        }
    }
}

/// `seconds` of audio, in samples.
#[must_use]
pub fn samples(seconds: f32) -> usize {
    (SAMPLE_RATE as f32 * seconds) as usize
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn limits(input: InputLimit) -> WindowLimits {
        WindowLimits {
            input,
            window: Duration::from_secs(30),
        }
    }

    #[test]
    fn a_whole_source_is_one_window_only_for_a_decoder_without_limit() {
        assert_eq!(
            WindowPolicy::choose(Windowing::Whole, limits(InputLimit::Unbounded), true, 64),
            WindowPolicy::Whole
        );
        assert_eq!(
            WindowPolicy::choose(
                Windowing::Whole,
                limits(InputLimit::UpTo(Duration::from_secs(20))),
                false,
                64
            ),
            WindowPolicy::Full { max: 320_000 }
        );
    }

    #[test]
    fn a_stream_with_a_detector_cuts_at_pauses_and_without_waits_for_full_windows() {
        assert_eq!(
            WindowPolicy::choose(Windowing::Stream, limits(InputLimit::Unbounded), true, 64),
            WindowPolicy::AtPauses {
                step: 64,
                max: 480_000
            }
        );
        assert_eq!(
            WindowPolicy::choose(Windowing::Stream, limits(InputLimit::Unbounded), false, 64),
            WindowPolicy::Full { max: 480_000 }
        );
    }

    #[test]
    fn whisper_margins_stay_as_they_were() {
        assert_eq!(
            Margins::default(),
            Margins {
                min: 1_600,
                min_paused: 16_000,
                commit: 16_000,
                retry_floor: 80_000,
            }
        );
    }
}
