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
    /// More audio will arrive, but the window has reached its maximum length.
    Full,
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
/// committed only when no closed segment lets the window advance.
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
        WindowKind::Last => Commit {
            segments: spans.len(),
            advance: len,
        },
        WindowKind::Full if spans.is_empty() => Commit {
            segments: 0,
            advance: horizon,
        },
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
            commit(&spans, 10 * S, S, WindowKind::Full),
            Commit {
                segments: 2,
                advance: span(0.0, 6.5).end,
            }
        );
    }

    #[test]
    fn never_commits_an_unclosed_segment_before_the_window_is_last() {
        let spans = [span(0.0, 2.0), span(2.0, 30.0)];
        assert_eq!(
            commit(&spans, 10 * S, S, WindowKind::Full),
            Commit {
                segments: 1,
                advance: 2 * S,
            }
        );
    }

    #[test]
    fn drops_speechless_audio_up_to_the_margin() {
        assert_eq!(
            commit(&[], 10 * S, S, WindowKind::Full),
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
