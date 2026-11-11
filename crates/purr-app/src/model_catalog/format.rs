use super::activity::DownloadProgress;
use crate::utils::byte_size::ByteSize;

/// `1.2 MiB/s`.
#[must_use]
pub fn speed(bytes_per_second: u64) -> String {
    format!("{}/s", ByteSize(bytes_per_second))
}

/// `42 s left`, `3 min left` or `1 h 5 min left`.
#[must_use]
pub fn time_left(seconds: u64) -> String {
    if seconds < 60 {
        format!("{seconds} s left")
    } else if seconds < 3_600 {
        format!("{} min left", seconds.div_ceil(60))
    } else {
        let minutes = seconds.div_ceil(60);
        format!("{} h {} min left", minutes / 60, minutes % 60)
    }
}

/// `37.2 MiB of 116.6 MiB`, or only the received size while the total is unknown.
#[must_use]
pub fn transferred(progress: &DownloadProgress) -> String {
    progress.total_bytes.map_or_else(
        || ByteSize(progress.loaded_bytes).to_string(),
        |total| format!("{} of {}", ByteSize(progress.loaded_bytes), ByteSize(total)),
    )
}

/// `1.2 MiB/s · 42 s left` with whatever part is known; empty before the first measurement.
#[must_use]
pub fn rate_and_eta(progress: &DownloadProgress) -> String {
    let parts: Vec<String> = [
        progress.bytes_per_second.map(speed),
        progress.seconds_left().map(time_left),
    ]
    .into_iter()
    .flatten()
    .collect();
    parts.join(" · ")
}

/// The whole-number percentage, `None` while the total is unknown.
#[must_use]
pub fn percent(progress: &DownloadProgress) -> Option<u32> {
    progress
        .fraction()
        .map(|fraction| (fraction * 100.0).round() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_left_switches_units() {
        assert_eq!(time_left(0), "0 s left");
        assert_eq!(time_left(59), "59 s left");
        assert_eq!(time_left(60), "1 min left");
        assert_eq!(time_left(61), "2 min left");
        assert_eq!(time_left(3_599), "60 min left");
        assert_eq!(time_left(3_900), "1 h 5 min left");
    }

    #[test]
    fn transferred_names_both_sizes_in_binary_units() {
        let progress = DownloadProgress {
            loaded_bytes: 39_000_000,
            total_bytes: Some(122_309_681),
            bytes_per_second: None,
        };
        assert_eq!(transferred(&progress), "37.2 MiB of 116.6 MiB");
        let unknown = DownloadProgress {
            loaded_bytes: 2_048,
            ..DownloadProgress::default()
        };
        assert_eq!(transferred(&unknown), "2.0 KiB");
    }

    #[test]
    fn rate_and_eta_show_what_is_known() {
        let mut progress = DownloadProgress {
            loaded_bytes: 0,
            total_bytes: Some(10 * 1024 * 1024),
            bytes_per_second: None,
        };
        assert_eq!(rate_and_eta(&progress), "");
        progress.bytes_per_second = Some(1024 * 1024);
        assert_eq!(rate_and_eta(&progress), "1.0 MiB/s · 10 s left");
    }

    #[test]
    fn percent_rounds_to_whole_numbers() {
        let progress = DownloadProgress {
            loaded_bytes: 1,
            total_bytes: Some(3),
            bytes_per_second: None,
        };
        assert_eq!(percent(&progress), Some(33));
        assert_eq!(percent(&DownloadProgress::default()), None);
    }
}
