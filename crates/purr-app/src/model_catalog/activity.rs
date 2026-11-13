use purr_common::platform::{IntegrityProblem, ModelFailureKind};
use std::collections::VecDeque;

/// How far a download has come.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DownloadProgress {
    pub loaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub bytes_per_second: Option<u64>,
}

impl DownloadProgress {
    /// The finished fraction between 0 and 1, or `None` while the total is unknown.
    #[must_use]
    pub fn fraction(&self) -> Option<f64> {
        let total = self.total_bytes.filter(|total| *total > 0)?;
        Some((self.loaded_bytes as f64 / total as f64).clamp(0.0, 1.0))
    }

    /// Whole seconds until the download ends at the current speed.
    #[must_use]
    pub fn seconds_left(&self) -> Option<u64> {
        let total = self.total_bytes?;
        let speed = self.bytes_per_second.filter(|speed| *speed > 0)?;
        Some(total.saturating_sub(self.loaded_bytes).div_ceil(speed))
    }
}

/// How far the check of a model's stored files has come.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VerifyProgress {
    pub checked_bytes: u64,
    pub total_bytes: Option<u64>,
}

impl VerifyProgress {
    /// The finished fraction between 0 and 1, or `None` while the total is unknown.
    #[must_use]
    pub fn fraction(&self) -> Option<f64> {
        let total = self.total_bytes.filter(|total| *total > 0)?;
        Some((self.checked_bytes as f64 / total as f64).clamp(0.0, 1.0))
    }
}

/// Why a download did not finish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadFailure {
    pub kind: ModelFailureKind,
    pub message: String,
}

/// What a check of the stored files found wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Damage {
    pub problem: IntegrityProblem,
    pub file: String,
}

/// Something happening to a model right now, or that went wrong and awaits the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelActivity {
    Downloading(DownloadProgress),
    Verifying(VerifyProgress),
    Failed(DownloadFailure),
    Damaged(Damage),
}

const WINDOW_MS: f64 = 5_000.0;
const MIN_SPAN_MS: f64 = 400.0;

/// Measures download speed over the last few seconds of progress reports.
#[derive(Debug, Clone, Default)]
pub struct SpeedMeter {
    samples: VecDeque<(f64, u64)>,
}

impl SpeedMeter {
    /// Records that `bytes` had arrived in total at `now_ms` and returns the speed in bytes per second.
    pub fn record(&mut self, now_ms: f64, bytes: u64) -> Option<u64> {
        self.samples.push_back((now_ms, bytes));
        while self
            .samples
            .front()
            .is_some_and(|(at, _)| now_ms - at > WINDOW_MS)
            && self.samples.len() > 2
        {
            self.samples.pop_front();
        }
        let (first_at, first_bytes) = *self.samples.front()?;
        let span = now_ms - first_at;
        if span < MIN_SPAN_MS {
            return None;
        }
        let gained = bytes.saturating_sub(first_bytes);
        Some((gained as f64 * 1000.0 / span) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fraction_needs_a_known_total() {
        let unknown = DownloadProgress {
            loaded_bytes: 10,
            ..DownloadProgress::default()
        };
        assert_eq!(unknown.fraction(), None);
        let half = DownloadProgress {
            loaded_bytes: 50,
            total_bytes: Some(100),
            bytes_per_second: None,
        };
        assert_eq!(half.fraction(), Some(0.5));
    }

    #[test]
    fn a_check_needs_a_known_total_to_have_a_fraction() {
        let unknown = VerifyProgress {
            checked_bytes: 10,
            total_bytes: None,
        };
        assert_eq!(unknown.fraction(), None);
        let quarter = VerifyProgress {
            checked_bytes: 25,
            total_bytes: Some(100),
        };
        assert_eq!(quarter.fraction(), Some(0.25));
        let empty = VerifyProgress {
            checked_bytes: 0,
            total_bytes: Some(0),
        };
        assert_eq!(empty.fraction(), None);
    }

    #[test]
    fn fraction_never_leaves_the_unit_range() {
        let over = DownloadProgress {
            loaded_bytes: 150,
            total_bytes: Some(100),
            bytes_per_second: None,
        };
        assert_eq!(over.fraction(), Some(1.0));
    }

    #[test]
    fn time_left_rounds_up_and_needs_a_speed() {
        let progress = DownloadProgress {
            loaded_bytes: 0,
            total_bytes: Some(1_000),
            bytes_per_second: Some(300),
        };
        assert_eq!(progress.seconds_left(), Some(4));
        let stalled = DownloadProgress {
            bytes_per_second: Some(0),
            ..progress
        };
        assert_eq!(stalled.seconds_left(), None);
        let finished = DownloadProgress {
            loaded_bytes: 1_000,
            ..progress
        };
        assert_eq!(finished.seconds_left(), Some(0));
    }

    #[test]
    fn the_meter_waits_for_a_usable_span() {
        let mut meter = SpeedMeter::default();
        assert_eq!(meter.record(0.0, 0), None);
        assert_eq!(meter.record(100.0, 1_000), None);
        assert_eq!(meter.record(1_000.0, 10_000), Some(10_000));
    }

    #[test]
    fn the_meter_forgets_samples_older_than_the_window() {
        let mut meter = SpeedMeter::default();
        meter.record(0.0, 0);
        meter.record(1_000.0, 1_000_000);
        meter.record(7_000.0, 1_000_100);
        assert_eq!(meter.record(8_000.0, 1_000_200), Some(100));
    }
}
