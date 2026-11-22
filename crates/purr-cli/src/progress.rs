//! The words of the progress status row, as pure functions of what is known.

use console::{measure_text_width, truncate_str};

const SEPARATOR: &str = "  ";
const NAME_FLOOR: usize = 8;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchStatus {
    pub index: usize,
    pub total: usize,
    pub done: usize,
    pub failed: usize,
    pub skipped: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StatusLength {
    pub seconds: f64,
    pub estimated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Status<'a> {
    pub name: &'a str,
    pub position: f64,
    pub length: Option<StatusLength>,
    pub batch: Option<BatchStatus>,
    pub speed: Option<f64>,
    pub elapsed: f64,
    pub eta: Option<f64>,
    pub width: usize,
}

/// The status row without its spinner; the name is shortened when the row would not fit `width`.
#[must_use]
pub fn status_text(status: &Status<'_>) -> String {
    let prefix = status.batch.map_or_else(String::new, |batch| {
        format!("[{}/{}] ", batch.index, batch.total)
    });

    let mut rest = vec![match status.length {
        Some(StatusLength { seconds, estimated }) => format!(
            "{} / {}{}",
            clock(status.position),
            if estimated { "~" } else { "" },
            clock(seconds)
        ),
        None => clock(status.position),
    }];
    if let Some(batch) = status.batch {
        rest.push(format!(
            "{} done, {} failed, {} skipped",
            batch.done, batch.failed, batch.skipped
        ));
    }
    if let Some(speed) = status.speed {
        rest.push(format!("{speed:.1}x"));
    }
    rest.push(clock(status.elapsed));
    if let Some(eta) = status.eta {
        rest.push(format!("ETA {}", clock(eta)));
    }
    let rest = rest.join(SEPARATOR);

    let taken = measure_text_width(&prefix) + SEPARATOR.len() + measure_text_width(&rest);
    let budget = status.width.saturating_sub(taken).max(NAME_FLOOR);
    let name = truncate_str(status.name, budget, "…");
    format!("{prefix}{name}{SEPARATOR}{rest}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn clocks_read_minutes_then_hours() {
        assert_eq!(clock(0.0), "0:00");
        assert_eq!(clock(171.9), "2:51");
        assert_eq!(clock(450.0), "7:30");
        assert_eq!(clock(3_725.0), "1:02:05");
        assert_eq!(clock(-3.0), "0:00");
    }

    fn status(name: &str) -> Status<'_> {
        Status {
            name,
            position: 11.0,
            length: Some(StatusLength {
                seconds: 27.0,
                estimated: false,
            }),
            batch: None,
            speed: Some(2.75),
            elapsed: 4.0,
            eta: Some(6.0),
            width: 80,
        }
    }

    #[test]
    fn a_single_input_shows_position_length_speed_elapsed_and_eta() {
        assert_eq!(
            status_text(&status("jfk.wav")),
            "jfk.wav  0:11 / 0:27  2.8x  0:04  ETA 0:06"
        );
    }

    #[test]
    fn an_unknown_length_shows_the_position_alone_and_no_eta() {
        let unknown = Status {
            length: None,
            eta: None,
            speed: None,
            ..status("radio")
        };
        assert_eq!(status_text(&unknown), "radio  0:11  0:04");
    }

    #[test]
    fn an_estimated_length_is_marked() {
        let estimated = Status {
            length: Some(StatusLength {
                seconds: 27.0,
                estimated: true,
            }),
            ..status("a.mp3")
        };
        assert_eq!(
            status_text(&estimated),
            "a.mp3  0:11 / ~0:27  2.8x  0:04  ETA 0:06"
        );
    }

    #[test]
    fn a_batch_shows_the_index_and_the_counts() {
        let batch = Status {
            batch: Some(BatchStatus {
                index: 2,
                total: 3,
                done: 1,
                failed: 0,
                skipped: 1,
            }),
            ..status("b.wav")
        };
        assert_eq!(
            status_text(&batch),
            "[2/3] b.wav  0:11 / 0:27  1 done, 0 failed, 1 skipped  2.8x  0:04  ETA 0:06"
        );
    }

    #[test]
    fn a_long_name_is_shortened_to_fit_the_width() {
        let narrow = Status {
            width: 50,
            ..status("a-very-long-recording-name-from-last-week.wav")
        };
        let text = status_text(&narrow);
        assert_eq!(text, "a-very-long-re…  0:11 / 0:27  2.8x  0:04  ETA 0:06");
        assert_eq!(measure_text_width(&text), 50);
    }
}
