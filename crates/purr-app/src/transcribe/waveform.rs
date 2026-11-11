//! The waveform the player draws: one bar per slice of the decoded audio.

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

/// Bars per waveform: enough for a wide player, few enough to stay light.
pub const BARS: usize = 120;

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
}
