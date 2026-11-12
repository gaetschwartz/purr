//! The rows the transcript card shows: the committed segments, then the tentative tail of a run.

use purr_common::platform::TranscriptSegment;

/// One row of the transcript card.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Row<'a> {
    /// A segment of the transcript, by its position
    Committed(usize, &'a TranscriptSegment),
    /// The uncommitted end of the stream, until the next update replaces it
    Tentative(&'a TranscriptSegment),
}

/// The rows to show, the tentative one last and only while the run is going.
#[must_use]
pub fn rows<'a>(
    segments: &'a [TranscriptSegment],
    tentative: Option<&'a TranscriptSegment>,
    live: bool,
) -> Vec<Row<'a>> {
    segments
        .iter()
        .enumerate()
        .map(|(index, segment)| Row::Committed(index, segment))
        .chain(tentative.filter(|_| live).map(Row::Tentative))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(text: &str) -> TranscriptSegment {
        TranscriptSegment {
            text: text.to_owned(),
            start_time: 0.0,
            end_time: 1.0,
            words: Vec::new(),
        }
    }

    #[test]
    fn the_tentative_row_comes_last_while_the_run_goes() {
        let committed = [segment("one"), segment("two")];
        let tail = segment("thr");
        assert_eq!(
            rows(&committed, Some(&tail), true),
            vec![
                Row::Committed(0, &committed[0]),
                Row::Committed(1, &committed[1]),
                Row::Tentative(&tail),
            ]
        );
    }

    #[test]
    fn the_tentative_row_never_outlives_the_run() {
        let committed = [segment("one")];
        let tail = segment("two");
        assert_eq!(
            rows(&committed, Some(&tail), false),
            vec![Row::Committed(0, &committed[0])]
        );
    }

    #[test]
    fn a_run_may_show_only_its_tentative_text() {
        let tail = segment("one");
        assert_eq!(rows(&[], Some(&tail), true), vec![Row::Tentative(&tail)]);
        assert_eq!(rows(&[], None, true), Vec::<Row<'_>>::new());
    }
}
