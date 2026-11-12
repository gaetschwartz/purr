//! Durations, positions and speeds as the Transcribe screens write them.

/// `m:ss` for a position or length of audio, `h:mm:ss` from an hour on.
#[must_use]
pub fn clock(seconds: f64) -> String {
    let total = seconds.max(0.0).floor() as u64;
    let (hours, minutes, secs) = (total / 3600, (total / 60) % 60, total % 60);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{secs:02}")
    } else {
        format!("{minutes}:{secs:02}")
    }
}

/// How long a run took: `24 s`, `6 min 42 s`, `2 min`.
#[must_use]
pub fn took(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    if total < 60 {
        return format!("{total} s");
    }
    let (minutes, secs) = (total / 60, total % 60);
    if secs == 0 {
        format!("{minutes} min")
    } else {
        format!("{minutes} min {secs} s")
    }
}

/// How many seconds of audio one second of work covers; `None` until both are measurable.
#[must_use]
pub fn speed(audio_seconds: f64, elapsed_seconds: f64) -> Option<f64> {
    (audio_seconds > 0.0 && elapsed_seconds > 0.0).then(|| audio_seconds / elapsed_seconds)
}

/// A speed as a multiple of real time: `4.6×`.
#[must_use]
pub fn speed_text(speed: f64) -> String {
    format!("{speed:.1}×")
}

/// Seconds left at `speed` for the audio not transcribed yet.
#[must_use]
pub fn seconds_left(audio_seconds: f64, done_seconds: f64, speed: f64) -> Option<f64> {
    (speed > 0.0 && audio_seconds > done_seconds).then(|| (audio_seconds - done_seconds) / speed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_pads_seconds_and_adds_hours_when_needed() {
        assert_eq!(clock(0.0), "0:00");
        assert_eq!(clock(75.9), "1:15");
        assert_eq!(clock(-3.0), "0:00");
        assert_eq!(clock(3_725.0), "1:02:05");
    }

    #[test]
    fn took_reads_naturally() {
        assert_eq!(took(24.4), "24 s");
        assert_eq!(took(402.0), "6 min 42 s");
        assert_eq!(took(120.0), "2 min");
    }

    #[test]
    fn speed_needs_both_measurements() {
        assert_eq!(speed(112.0, 24.0).map(speed_text), Some("4.7×".to_owned()));
        assert_eq!(speed(0.0, 3.0), None);
        assert_eq!(speed(10.0, 0.0), None);
    }

    #[test]
    fn time_left_follows_the_speed() {
        assert_eq!(seconds_left(100.0, 40.0, 2.0), Some(30.0));
        assert_eq!(seconds_left(100.0, 100.0, 2.0), None);
        assert_eq!(seconds_left(100.0, 40.0, 0.0), None);
    }
}
