//! Time left in a run: a long-window rate, rounded words, and a display that never chatters.
//!
//! All times are in seconds. Points are `(wall, transcribed)` pairs, oldest first, starting with
//! a synthetic `(start, 0)`.

use std::fmt;

/// The rate is measured over at least this much wall time once there are enough updates.
pub const WINDOW: f64 = 30.0;
/// Updates needed, besides the start point, before there is an estimate.
pub const MIN_UPDATES: usize = 4;
/// Wall time the updates must span before there is an estimate.
pub const MIN_SPAN: f64 = 10.0;
/// The text changes at most this often.
pub const HOLD: f64 = 5.0;
/// An increase is shown only above this multiple of what is shown…
pub const UP_RATIO: f64 = 1.25;
/// …for this long in a row.
pub const UP_HOLD: f64 = 10.0;
/// Below this the words are "almost done".
pub const ALMOST: f64 = 10.0;
/// Estimates longer than this are not shown.
pub const MAX: f64 = 12.0 * 3600.0;

/// The rounded time left, as the run view words it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EtaText {
    AlmostDone,
    Seconds(u8),
    Minutes(u16),
    HoursMinutes(u8, u8),
}

impl EtaText {
    /// The seconds the text stands for, to compare buckets.
    #[must_use]
    pub fn seconds(self) -> f64 {
        match self {
            Self::AlmostDone => 0.0,
            Self::Seconds(seconds) => f64::from(seconds),
            Self::Minutes(minutes) => f64::from(minutes) * 60.0,
            Self::HoursMinutes(hours, minutes) => {
                f64::from(hours) * 3600.0 + f64::from(minutes) * 60.0
            }
        }
    }
}

impl fmt::Display for EtaText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlmostDone => f.write_str("almost done"),
            Self::Seconds(seconds) => write!(f, "about {seconds} s left"),
            Self::Minutes(minutes) => write!(f, "about {minutes} min left"),
            Self::HoursMinutes(hours, 0) => write!(f, "about {hours} h left"),
            Self::HoursMinutes(hours, minutes) => write!(f, "about {hours} h {minutes} min left"),
        }
    }
}

/// Seconds of audio per wall second over the window; none before enough updates and time.
#[must_use]
pub fn eta_rate(history: &[(f64, f64)]) -> Option<f64> {
    if history.len() < MIN_UPDATES + 1 {
        return None;
    }
    let (&(first_at, first), &(last_at, last)) = (history.first()?, history.last()?);
    let span = last_at - first_at;
    (span >= MIN_SPAN).then(|| (last - first) / span)
}

/// Wall seconds left, counting down since the last update; none without a usable rate.
#[must_use]
pub fn eta_seconds(remaining_audio: f64, rate: f64, since_update: f64) -> Option<f64> {
    if rate <= 0.0 || remaining_audio < 0.0 {
        return None;
    }
    let seconds = (remaining_audio / rate - since_update).max(0.0);
    (seconds <= MAX).then_some(seconds)
}

/// The words for `seconds` left.
#[must_use]
pub fn eta_bucket(seconds: f64) -> EtaText {
    if seconds < ALMOST {
        EtaText::AlmostDone
    } else if seconds < 50.0 {
        EtaText::Seconds(((seconds / 10.0).ceil() * 10.0) as u8)
    } else if seconds < 570.0 {
        EtaText::Minutes((seconds / 60.0).round().max(1.0) as u16)
    } else if seconds < 3600.0 {
        let minutes = (seconds / 300.0).round() as u16 * 5;
        if minutes >= 60 {
            EtaText::HoursMinutes(1, 0)
        } else {
            EtaText::Minutes(minutes)
        }
    } else {
        let tens = (seconds / 600.0).round() as u16;
        let total = tens * 10;
        EtaText::HoursMinutes((total / 60) as u8, (total % 60) as u8)
    }
}

/// What is on screen and since when.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shown {
    pub text: EtaText,
    /// When the text last changed.
    pub at: f64,
    /// Since when the raw value has been well above the text, if it has.
    pub up_since: Option<f64>,
}

/// The text to show at `now` for the raw estimate `raw`, given what is shown.
///
/// Changes wait for the hold; increases show only after staying well above for a while.
#[must_use]
pub fn eta_display(previous: Option<Shown>, now: f64, raw: Option<f64>) -> Option<Shown> {
    let raw = raw?;
    let bucket = eta_bucket(raw);
    let Some(previous) = previous else {
        return Some(Shown {
            text: bucket,
            at: now,
            up_since: None,
        });
    };
    let fresh = Shown {
        text: bucket,
        at: now,
        up_since: None,
    };
    let calm = Shown {
        up_since: None,
        ..previous
    };
    let shown = previous.text.seconds();
    if bucket.seconds() == shown {
        return Some(calm);
    }
    if now - previous.at < HOLD {
        return Some(previous);
    }
    if bucket.seconds() < shown {
        return Some(fresh);
    }
    if raw <= shown.max(ALMOST) * UP_RATIO {
        return Some(calm);
    }
    let since = previous.up_since.unwrap_or(now);
    Some(if now - since >= UP_HOLD {
        fresh
    } else {
        Shown {
            up_since: Some(since),
            ..previous
        }
    })
}

/// The time left of one run, fed with each new transcribed position.
#[derive(Debug, Clone, PartialEq)]
pub struct EtaTracker {
    history: Vec<(f64, f64)>,
    last_update: f64,
    shown: Option<Shown>,
}

impl EtaTracker {
    /// A run that starts transcribing at `now` with `transcribed` seconds already done.
    #[must_use]
    pub fn new(now: f64, transcribed: f64) -> Self {
        Self {
            history: vec![(now, transcribed)],
            last_update: now,
            shown: None,
        }
    }

    /// A new transcribed position at `now`.
    pub fn update(&mut self, now: f64, transcribed: f64) {
        self.history.push((now, transcribed));
        self.last_update = now;
        while self.history.len() > MIN_UPDATES + 1
            && self
                .history
                .get(1)
                .is_some_and(|&(at, _)| now - at >= WINDOW)
        {
            self.history.remove(0);
        }
    }

    /// The text at `now` with `remaining_audio` seconds of audio left.
    pub fn tick(&mut self, now: f64, remaining_audio: f64) -> Option<EtaText> {
        let rate = eta_rate(&self.history);
        let raw = rate.and_then(|rate| eta_seconds(remaining_audio, rate, now - self.last_update));
        self.shown = match (raw, rate) {
            (None, Some(_)) => self.shown,
            (None, None) => None,
            (Some(_), _) => eta_display(self.shown, now, raw),
        };
        self.shown.map(|shown| shown.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shown(text: EtaText, at: f64) -> Option<Shown> {
        Some(Shown {
            text,
            at,
            up_since: None,
        })
    }

    #[test]
    fn the_rate_waits_for_four_updates_and_ten_seconds() {
        assert_eq!(
            eta_rate(&[(0.0, 0.0), (3.0, 4.0), (6.0, 8.0), (9.0, 12.0)]),
            None
        );
        assert_eq!(
            eta_rate(&[(0.0, 0.0), (1.0, 4.0), (2.0, 8.0), (3.0, 12.0), (4.0, 16.0)]),
            None
        );
        let rate = eta_rate(&[
            (0.0, 0.0),
            (3.0, 4.0),
            (6.0, 8.0),
            (9.0, 12.0),
            (12.0, 16.0),
        ]);
        assert!((rate.unwrap() - 4.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn the_time_left_counts_down_between_updates() {
        assert!((eta_seconds(40.0, 4.0 / 3.0, 0.0).unwrap() - 30.0).abs() < 1e-9);
        assert!((eta_seconds(40.0, 4.0 / 3.0, 2.0).unwrap() - 28.0).abs() < 1e-9);
        assert_eq!(eta_seconds(40.0, 0.0, 0.0), None);
        assert_eq!(eta_seconds(13.0 * 3600.0, 1.0, 0.0), None);
    }

    #[test]
    fn buckets_round_as_the_spec_says() {
        assert_eq!(eta_bucket(9.9), EtaText::AlmostDone);
        assert_eq!(eta_bucket(10.0), EtaText::Seconds(10));
        assert_eq!(eta_bucket(41.0), EtaText::Seconds(50));
        assert_eq!(eta_bucket(56.0), EtaText::Minutes(1));
        assert_eq!(eta_bucket(95.0), EtaText::Minutes(2));
        assert_eq!(eta_bucket(600.0), EtaText::Minutes(10));
        assert_eq!(eta_bucket(1720.0), EtaText::Minutes(30));
        assert_eq!(eta_bucket(3599.0), EtaText::HoursMinutes(1, 0));
        assert_eq!(eta_bucket(4900.0), EtaText::HoursMinutes(1, 20));
    }

    #[test]
    fn the_text_holds_for_five_seconds() {
        let previous = shown(EtaText::Minutes(4), 100.0);
        assert_eq!(eta_display(previous, 103.0, Some(120.0)), previous);
    }

    #[test]
    fn a_small_rise_is_never_shown() {
        let mut display = shown(EtaText::Minutes(4), 0.0);
        for second in 10..60 {
            display = eta_display(display, f64::from(second), Some(240.0 * 1.13));
            assert_eq!(display.map(|shown| shown.text), Some(EtaText::Minutes(4)));
        }
    }

    #[test]
    fn a_large_rise_shows_only_after_ten_seconds_in_a_row() {
        let mut display = shown(EtaText::Minutes(4), 0.0);
        for second in 10..20 {
            display = eta_display(display, f64::from(second), Some(240.0 * 1.73));
            assert_eq!(display.map(|shown| shown.text), Some(EtaText::Minutes(4)));
        }
        display = eta_display(display, 20.0, Some(240.0 * 1.73));
        assert_eq!(display.map(|shown| shown.text), Some(EtaText::Minutes(7)));
    }

    #[test]
    fn decreases_show_once_the_hold_has_passed() {
        let previous = shown(EtaText::Minutes(4), 0.0);
        assert_eq!(
            eta_display(previous, 4.0, Some(150.0)).map(|shown| shown.text),
            Some(EtaText::Minutes(4))
        );
        assert_eq!(
            eta_display(previous, 5.0, Some(150.0)),
            shown(EtaText::Minutes(3), 5.0)
        );
    }

    #[test]
    fn a_steady_run_shows_a_falling_estimate() {
        let mut tracker = EtaTracker::new(0.0, 0.0);
        assert_eq!(tracker.tick(1.0, 100.0), None);
        let mut done = 0.0;
        for update in 1..=5 {
            done += 4.0;
            tracker.update(f64::from(update) * 3.0, done);
        }
        assert_eq!(tracker.tick(15.0, 100.0 - done), Some(EtaText::Minutes(1)));
    }
}
