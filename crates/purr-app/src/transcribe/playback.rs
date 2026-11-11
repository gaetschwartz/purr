//! Where the player is in the transcript.

use purr_common::platform::TranscriptSegment;

/// The segment playing at `position` seconds: the last one that started at or before it.
#[must_use]
pub fn segment_at(segments: &[TranscriptSegment], position: f64) -> Option<usize> {
    segments
        .iter()
        .rposition(|segment| segment.start_time <= position)
}

/// The position `step` seconds away from `position`, kept within the audio.
#[must_use]
pub fn step(position: f64, step: f64, duration: f64) -> f64 {
    (position + step).clamp(0.0, duration.max(0.0))
}

/// The position at `fraction` of the audio, for a click on the waveform.
#[must_use]
pub fn at_fraction(fraction: f64, duration: f64) -> f64 {
    fraction.clamp(0.0, 1.0) * duration.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments() -> Vec<TranscriptSegment> {
        [0.0, 7.0, 15.0]
            .into_iter()
            .map(|start| TranscriptSegment {
                text: String::new(),
                start_time: start,
                end_time: start + 5.0,
            })
            .collect()
    }

    #[test]
    fn the_playing_segment_is_the_last_that_started() {
        let segments = segments();
        assert_eq!(segment_at(&segments, 0.0), Some(0));
        assert_eq!(segment_at(&segments, 6.9), Some(0));
        assert_eq!(segment_at(&segments, 7.0), Some(1));
        assert_eq!(segment_at(&segments, 99.0), Some(2));
    }

    #[test]
    fn nothing_plays_before_the_first_segment() {
        let late = vec![TranscriptSegment {
            text: String::new(),
            start_time: 2.0,
            end_time: 3.0,
        }];
        assert_eq!(segment_at(&late, 1.0), None);
        assert_eq!(segment_at(&[], 1.0), None);
    }

    #[test]
    fn steps_stay_within_the_audio() {
        assert_eq!(step(3.0, -5.0, 60.0), 0.0);
        assert_eq!(step(58.0, 5.0, 60.0), 60.0);
        assert_eq!(step(10.0, 5.0, 60.0), 15.0);
    }

    #[test]
    fn a_fraction_maps_onto_the_duration() {
        assert_eq!(at_fraction(0.5, 120.0), 60.0);
        assert_eq!(at_fraction(1.4, 120.0), 120.0);
        assert_eq!(at_fraction(-0.1, 120.0), 0.0);
    }
}
