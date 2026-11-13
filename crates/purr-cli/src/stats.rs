//! The statistics block shown after a transcription.

use crate::statusln;
use crate::theme::Themed as _;
use purr_core::whisper::TranscriptionStats;

/// The block's lines; `acceleration` is only known for a transcription that was not streamed.
pub fn lines(stats: &TranscriptionStats, acceleration: Option<&str>) -> Vec<String> {
    let mut lines = vec![
        format!("Audio duration: {:.2}s", stats.audio_duration),
        format!("Processing time: {:.2}s", stats.processing_time),
        format!("Real-time factor: {:.2}x", stats.real_time_factor()),
        format!("Segments: {}", stats.segment_count),
        format!("Words: {}", stats.word_count),
        format!("Words per minute: {:.1}", stats.words_per_minute()),
    ];
    if let Some(acceleration) = acceleration {
        lines.push(format!("Acceleration: {acceleration}"));
    }
    lines
}

pub fn show(title: &str, stats: &TranscriptionStats, acceleration: Option<&str>) {
    statusln!();
    statusln!("{}", title.heading());
    for line in lines(stats, acceleration) {
        statusln!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_block_lists_each_figure_and_the_acceleration_when_known() {
        let stats = TranscriptionStats {
            audio_duration: 11.0,
            processing_time: 2.0,
            segment_count: 3,
            word_count: 22,
        };
        assert_eq!(
            lines(&stats, Some("CPU")),
            [
                "Audio duration: 11.00s",
                "Processing time: 2.00s",
                "Real-time factor: 5.50x",
                "Segments: 3",
                "Words: 22",
                "Words per minute: 120.0",
                "Acceleration: CPU",
            ]
        );
    }
}
