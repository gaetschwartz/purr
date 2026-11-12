//! The `--progress` line.

use purr_core::whisper::StreamProgress;

/// `seconds` as `m:ss`, or `h:mm:ss` from an hour on.
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

/// How far the stream has come; the duration shows once the source has delivered all its audio.
#[must_use]
pub fn line(progress: &StreamProgress) -> String {
    let done = clock(progress.frame.transcribed_to);
    if progress.source_ended {
        format!("{done} / {}", clock(progress.received))
    } else {
        done
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::progress::{ProgressFrame, ProgressPhase};

    #[test]
    fn clocks_read_minutes_then_hours() {
        assert_eq!(clock(0.0), "0:00");
        assert_eq!(clock(171.9), "2:51");
        assert_eq!(clock(450.0), "7:30");
        assert_eq!(clock(3_725.0), "1:02:05");
        assert_eq!(clock(-3.0), "0:00");
    }

    #[test]
    fn the_duration_shows_once_the_source_has_ended() {
        let mut progress = StreamProgress {
            frame: ProgressFrame {
                transcribed_to: 171.0,
                decoding: None,
                phase: ProgressPhase::Transcribing,
            },
            received: 200.0,
            source_ended: false,
        };
        assert_eq!(line(&progress), "2:51");
        progress.received = 450.0;
        progress.source_ended = true;
        assert_eq!(line(&progress), "2:51 / 7:30");
    }
}
