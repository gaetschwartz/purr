//! Which decoded segments of a streaming window are final, and where the next window starts.

/// A decoded segment's span, in samples from the start of its window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// How much audio may still follow a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowKind {
    /// More audio will arrive, and the window may grow before anything must be committed.
    Growing,
    /// More audio will arrive, but the window has reached its maximum length.
    Full,
    /// The window ends in a pause between words, so nothing in it can be cut.
    Paused,
    /// The stream has ended; nothing follows this window.
    Last,
}

/// The decision for one decoded window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Commit {
    /// How many leading segments are final
    pub segments: usize,
    /// How many samples of the window are consumed; the next window starts here
    pub advance: usize,
}

/// Commits every segment that Whisper closed before `margin` samples from the window's end, and
/// starts the next window at the end of the last committed segment.
///
/// A segment Whisper did not close ends 30 s after it starts, past the window, so it is
/// committed only when the window cannot grow and no closed segment lets it advance.
pub fn commit(spans: &[Span], len: usize, margin: usize, kind: WindowKind) -> Commit {
    let horizon = len.saturating_sub(margin);
    let through = |limit: usize| {
        spans
            .iter()
            .take_while(|span| span.end <= limit && span.start <= span.end)
            .enumerate()
            .last()
            .map(|(i, span)| Commit {
                segments: i + 1,
                advance: span.end,
            })
    };
    match kind {
        WindowKind::Last | WindowKind::Paused => Commit {
            segments: spans.len(),
            advance: len,
        },
        WindowKind::Growing | WindowKind::Full if spans.is_empty() => Commit {
            segments: 0,
            advance: horizon,
        },
        WindowKind::Growing => through(horizon).unwrap_or(Commit {
            segments: 0,
            advance: 0,
        }),
        WindowKind::Full => through(horizon).or_else(|| through(len)).unwrap_or(Commit {
            segments: spans.len(),
            advance: len,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: usize = 16_000;

    fn span(start: f32, end: f32) -> Span {
        Span {
            start: (start * S as f32) as usize,
            end: (end * S as f32) as usize,
        }
    }

    #[test]
    fn commits_segments_closed_before_the_margin() {
        let spans = [span(0.0, 3.0), span(3.0, 6.5), span(6.5, 9.8)];
        assert_eq!(
            commit(&spans, 10 * S, S, WindowKind::Growing),
            Commit {
                segments: 2,
                advance: span(0.0, 6.5).end,
            }
        );
    }

    #[test]
    fn holds_a_growing_window_whose_only_segment_reaches_its_end() {
        assert_eq!(
            commit(&[span(0.5, 9.9)], 10 * S, S, WindowKind::Growing),
            Commit {
                segments: 0,
                advance: 0,
            }
        );
    }

    #[test]
    fn never_commits_an_unclosed_segment_before_the_window_is_last() {
        let spans = [span(0.0, 2.0), span(2.0, 30.0)];
        assert_eq!(
            commit(&spans, 10 * S, S, WindowKind::Growing),
            Commit {
                segments: 1,
                advance: 2 * S,
            }
        );
    }

    #[test]
    fn drops_speechless_audio_up_to_the_margin() {
        assert_eq!(
            commit(&[], 10 * S, S, WindowKind::Growing),
            Commit {
                segments: 0,
                advance: 9 * S,
            }
        );
    }

    #[test]
    fn a_full_window_without_room_commits_its_segments_closed_inside_it() {
        let spans = [span(0.0, 29.8), span(29.8, 59.8)];
        assert_eq!(
            commit(&spans, 30 * S, S, WindowKind::Full),
            Commit {
                segments: 1,
                advance: span(0.0, 29.8).end,
            }
        );
    }

    #[test]
    fn a_full_window_with_one_unclosed_segment_commits_it_whole() {
        assert_eq!(
            commit(&[span(0.0, 60.0)], 30 * S, S, WindowKind::Full),
            Commit {
                segments: 1,
                advance: 30 * S,
            }
        );
    }

    #[test]
    fn a_window_ending_in_a_pause_commits_everything() {
        let spans = [span(0.0, 2.0), span(2.0, 4.6)];
        assert_eq!(
            commit(&spans, 5 * S, S, WindowKind::Paused),
            Commit {
                segments: 2,
                advance: 5 * S,
            }
        );
    }

    #[test]
    fn the_last_window_commits_everything() {
        let spans = [span(0.0, 2.0), span(2.0, 32.0)];
        assert_eq!(
            commit(&spans, 3 * S, S, WindowKind::Last),
            Commit {
                segments: 2,
                advance: 3 * S,
            }
        );
    }
}
