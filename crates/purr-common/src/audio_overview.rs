//! The length and waveform of a recording, measured by the platform's own decoder.

use serde::{Deserialize, Serialize};

/// What decoding a stored upload revealed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioOverview {
    pub duration_seconds: f64,
    /// Waveform bars between 0 and 1.
    pub peaks: Vec<f32>,
}

/// Windows per second that [`PeakFold`] keeps; the bars are drawn from these.
const WINDOWS_PER_SECOND: u32 = 100;

/// The loudest absolute sample of each of `buckets` equal slices, scaled so the loudest bar is 1.
///
/// Empty when there is no audio or no bucket; silence gives bars of 0.
#[must_use]
pub fn peaks(samples: &[f32], buckets: usize) -> Vec<f32> {
    if samples.is_empty() || buckets == 0 {
        return Vec::new();
    }
    let chunk = samples.len().div_ceil(buckets);
    let raw: Vec<f32> = samples
        .chunks(chunk)
        .map(|slice| {
            slice
                .iter()
                .fold(0.0_f32, |loudest, s| loudest.max(s.abs()))
        })
        .collect();
    let loudest = raw.iter().copied().fold(0.0_f32, f32::max);
    if loudest <= 0.0 {
        return raw;
    }
    raw.into_iter().map(|peak| peak / loudest).collect()
}

/// Folds decoded samples into a waveform as they stream by, keeping one value per 10 ms instead
/// of every sample, so a long recording never has to sit in memory whole.
#[derive(Debug, Clone)]
pub struct PeakFold {
    window: usize,
    filled: usize,
    loudest: f32,
    windows: Vec<f32>,
    total: u64,
}

impl PeakFold {
    /// A fold for audio at `sample_rate` hertz.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            window: (sample_rate / WINDOWS_PER_SECOND).max(1) as usize,
            filled: 0,
            loudest: 0.0,
            windows: Vec::new(),
            total: 0,
        }
    }

    /// Folds in the next samples.
    pub fn push(&mut self, samples: &[f32]) {
        self.total += samples.len() as u64;
        for sample in samples {
            self.loudest = self.loudest.max(sample.abs());
            self.filled += 1;
            if self.filled == self.window {
                self.windows.push(self.loudest);
                self.filled = 0;
                self.loudest = 0.0;
            }
        }
    }

    /// How many samples were folded in.
    #[must_use]
    pub const fn samples(&self) -> u64 {
        self.total
    }

    /// The `buckets` bars of everything folded in.
    #[must_use]
    pub fn finish(mut self, buckets: usize) -> Vec<f32> {
        if self.filled > 0 {
            self.windows.push(self.loudest);
        }
        peaks(&self.windows, buckets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_bucket_keeps_its_loudest_sample_scaled_to_one() {
        let samples = [0.1, -0.5, 0.25, 0.0, -0.2, 0.05];
        assert_eq!(peaks(&samples, 3), vec![1.0, 0.5, 0.4]);
    }

    #[test]
    fn fewer_samples_than_buckets_gives_one_bar_per_sample() {
        assert_eq!(peaks(&[0.5, -1.0], 10), vec![0.5, 1.0]);
    }

    #[test]
    fn silence_and_empty_input_are_handled() {
        assert_eq!(peaks(&[0.0; 8], 4), vec![0.0; 4]);
        assert_eq!(peaks(&[], 4), Vec::<f32>::new());
        assert_eq!(peaks(&[0.3], 0), Vec::<f32>::new());
    }

    #[test]
    fn uneven_lengths_still_cover_every_sample() {
        let samples = [0.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(peaks(&samples, 2), vec![0.0, 1.0]);
    }

    #[test]
    fn folding_in_pieces_matches_the_whole_when_windows_are_single_samples() {
        let samples = [0.1, -0.5, 0.25, 0.0, -0.2, 0.05, 0.9];
        let mut fold = PeakFold::new(100);
        fold.push(&samples[..3]);
        fold.push(&samples[3..4]);
        fold.push(&samples[4..]);
        assert_eq!(fold.samples(), 7);
        assert_eq!(fold.finish(3), peaks(&samples, 3));
    }

    #[test]
    fn a_window_may_span_two_pushes_and_a_partial_one_is_kept() {
        let mut fold = PeakFold::new(200);
        fold.push(&[0.1]);
        fold.push(&[0.4, 0.2]);
        fold.push(&[1.0]);
        assert_eq!(fold.finish(2), vec![0.4, 1.0]);
    }

    #[test]
    fn nothing_folded_gives_no_bars() {
        assert_eq!(PeakFold::new(16_000).finish(120), Vec::<f32>::new());
    }
}
