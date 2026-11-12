//! The transcript assembled from a stream of transcription updates.

use purr_common::platform::{ChunkKind, TranscriptSegment, TranscriptionStatus};
use std::collections::BTreeMap;

/// Segments reported so far for one transcription.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transcript {
    live: BTreeMap<usize, TranscriptSegment>,
    settled: Option<Vec<TranscriptSegment>>,
    /// The uncommitted end of the transcript and the index it waits at, kept out of `segments`
    tentative: Option<(usize, TranscriptSegment)>,
}

impl Transcript {
    /// Folds one status update into the transcript.
    pub fn apply(&mut self, status: &TranscriptionStatus) {
        match status {
            TranscriptionStatus::InProgress {
                chunk_index,
                text,
                start_time,
                end_time,
                words,
                kind,
            } => {
                let segment = TranscriptSegment {
                    text: text.clone(),
                    start_time: *start_time,
                    end_time: *end_time,
                    words: words.clone(),
                };
                match kind {
                    ChunkKind::Committed => {
                        if self
                            .tentative
                            .as_ref()
                            .is_some_and(|(index, _)| index <= chunk_index)
                        {
                            self.tentative = None;
                        }
                        self.live.insert(*chunk_index, segment);
                    }
                    ChunkKind::Tentative => self.tentative = Some((*chunk_index, segment)),
                }
            }
            TranscriptionStatus::Segments { segments } => {
                self.settled = Some(segments.clone());
            }
            TranscriptionStatus::BrowserEngine { .. } => {
                self.live.clear();
                self.tentative = None;
            }
            TranscriptionStatus::Completed { .. } => self.tentative = None,
            TranscriptionStatus::Starting
            | TranscriptionStatus::ProcessingAudio
            | TranscriptionStatus::LoadingModel { .. }
            | TranscriptionStatus::Transcribing
            | TranscriptionStatus::Engine { .. }
            | TranscriptionStatus::LanguageDetected { .. }
            | TranscriptionStatus::Progress(_)
            | TranscriptionStatus::Telemetry(_)
            | TranscriptionStatus::Error { .. }
            | TranscriptionStatus::InitFailed { .. } => {}
        }
    }

    /// The spoken segments: the final list once known, otherwise the live ones in order.
    #[must_use]
    pub fn segments(&self) -> Vec<TranscriptSegment> {
        let source: Vec<&TranscriptSegment> = self.settled.as_ref().map_or_else(
            || self.live.values().collect(),
            |settled| settled.iter().collect(),
        );
        source
            .into_iter()
            .filter(|segment| is_spoken(&segment.text))
            .cloned()
            .collect()
    }

    /// The uncommitted end of the transcript, shown early, when it holds words.
    #[must_use]
    pub fn tentative(&self) -> Option<&TranscriptSegment> {
        self.tentative
            .as_ref()
            .map(|(_, segment)| segment)
            .filter(|segment| is_spoken(&segment.text))
    }

    /// The whole transcript as one paragraph.
    #[must_use]
    pub fn text(&self) -> String {
        self.segments()
            .iter()
            .map(|segment| segment.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Whether a segment holds words, rather than nothing or a Whisper marker such as `[BLANK_AUDIO]`.
fn is_spoken(text: &str) -> bool {
    let text = text.trim();
    let marker = text
        .strip_prefix('[')
        .and_then(|inner| inner.strip_suffix(']'))
        .is_some_and(|inner| {
            !inner.trim().is_empty()
                && inner
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c == '_' || c == ' ')
        });
    !text.is_empty() && !marker
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::word::Word;

    fn in_progress(chunk_index: usize, text: &str, start: f64, end: f64) -> TranscriptionStatus {
        TranscriptionStatus::InProgress {
            chunk_index,
            text: text.to_owned(),
            start_time: start,
            end_time: end,
            words: Vec::new(),
            kind: ChunkKind::Committed,
        }
    }

    fn tentative(chunk_index: usize, text: &str, start: f64, end: f64) -> TranscriptionStatus {
        TranscriptionStatus::InProgress {
            chunk_index,
            text: text.to_owned(),
            start_time: start,
            end_time: end,
            words: Vec::new(),
            kind: ChunkKind::Tentative,
        }
    }

    #[test]
    fn tentative_text_stays_out_of_the_segments_and_the_next_one_replaces_it() {
        let mut transcript = Transcript::default();
        transcript.apply(&in_progress(0, " one", 0.0, 1.0));
        transcript.apply(&tentative(1, " two thr", 1.0, 2.0));
        transcript.apply(&tentative(1, " two three", 1.0, 3.0));
        assert_eq!(transcript.text(), "one");
        assert_eq!(
            transcript.tentative(),
            Some(&segment(" two three", 1.0, 3.0))
        );
    }

    #[test]
    fn a_committed_chunk_at_or_past_the_tentative_index_replaces_it() {
        let mut transcript = Transcript::default();
        transcript.apply(&tentative(1, " two three", 1.0, 3.0));
        transcript.apply(&in_progress(0, " one", 0.0, 1.0));
        assert_eq!(
            transcript.tentative(),
            Some(&segment(" two three", 1.0, 3.0))
        );
        transcript.apply(&in_progress(1, " two", 1.0, 2.0));
        assert_eq!(transcript.tentative(), None);
        assert_eq!(transcript.text(), "one two");

        transcript.apply(&tentative(2, " four", 3.0, 4.0));
        transcript.apply(&in_progress(3, " five", 4.0, 5.0));
        assert_eq!(transcript.tentative(), None);
    }

    #[test]
    fn completion_drops_tentative_text() {
        let mut transcript = Transcript::default();
        transcript.apply(&tentative(0, " draft", 0.0, 1.0));
        transcript.apply(&TranscriptionStatus::Completed {
            processing_time: 1.0,
            audio_duration: 1.0,
            word_count: 0,
        });
        assert_eq!(transcript.tentative(), None);
    }

    fn segment(text: &str, start: f64, end: f64) -> TranscriptSegment {
        TranscriptSegment {
            text: text.to_owned(),
            start_time: start,
            end_time: end,
            words: Vec::new(),
        }
    }

    #[test]
    fn growing_segment_replaces_itself_instead_of_repeating() {
        let mut transcript = Transcript::default();
        transcript.apply(&in_progress(0, " Et un", 0.0, 0.0));
        transcript.apply(&in_progress(0, " Et un grand bravo", 0.0, 2.0));
        assert_eq!(transcript.text(), "Et un grand bravo");
    }

    #[test]
    fn a_replaced_segment_takes_the_words_of_its_latest_text() {
        let word = |start: u32, end: u32, score: u8| Word {
            range: start..end,
            start_cs: Some(start * 10),
            end_cs: Some(end * 10),
            score: Some(score),
        };
        let mut transcript = Transcript::default();
        transcript.apply(&TranscriptionStatus::InProgress {
            chunk_index: 0,
            text: "Et".to_owned(),
            start_time: 0.0,
            end_time: 0.2,
            words: vec![word(0, 2, 90)],
            kind: ChunkKind::Committed,
        });
        transcript.apply(&TranscriptionStatus::InProgress {
            chunk_index: 0,
            text: "Et un".to_owned(),
            start_time: 0.0,
            end_time: 0.5,
            words: vec![word(0, 2, 200), word(3, 5, 180)],
            kind: ChunkKind::Committed,
        });
        assert_eq!(
            transcript.segments(),
            vec![TranscriptSegment {
                text: "Et un".to_owned(),
                start_time: 0.0,
                end_time: 0.5,
                words: vec![word(0, 2, 200), word(3, 5, 180)],
            }]
        );
    }

    #[test]
    fn segments_are_ordered_by_index() {
        let mut transcript = Transcript::default();
        transcript.apply(&in_progress(1, " world", 1.0, 2.0));
        transcript.apply(&in_progress(0, " hello", 0.0, 1.0));
        assert_eq!(transcript.text(), "hello world");
    }

    #[test]
    fn final_segments_replace_everything_live() {
        let mut transcript = Transcript::default();
        transcript.apply(&in_progress(0, " draft one", 0.0, 1.0));
        transcript.apply(&in_progress(1, " draft two", 1.0, 2.0));
        transcript.apply(&TranscriptionStatus::Segments {
            segments: vec![segment(" final", 0.0, 2.0)],
        });
        assert_eq!(transcript.segments(), vec![segment(" final", 0.0, 2.0)]);
    }

    #[test]
    fn blank_segments_are_dropped() {
        let mut transcript = Transcript::default();
        transcript.apply(&in_progress(0, "  ", 0.0, 1.0));
        assert_eq!(transcript.segments(), Vec::<TranscriptSegment>::new());
    }

    #[test]
    fn whisper_silence_markers_are_not_speech() {
        let mut transcript = Transcript::default();
        transcript.apply(&TranscriptionStatus::Segments {
            segments: vec![
                segment(" [BLANK_AUDIO]", 0.0, 1.0),
                segment(" [MUSIC PLAYING]", 1.0, 2.0),
                segment(" [Applause] thank you", 2.0, 3.0),
                segment(" [ ]", 3.0, 4.0),
            ],
        });
        assert_eq!(
            transcript.segments(),
            vec![
                segment(" [Applause] thank you", 2.0, 3.0),
                segment(" [ ]", 3.0, 4.0),
            ]
        );
    }

    #[test]
    fn other_statuses_leave_the_transcript_alone() {
        let mut transcript = Transcript::default();
        transcript.apply(&TranscriptionStatus::Transcribing);
        transcript.apply(&TranscriptionStatus::LoadingModel {
            loaded_bytes: 1,
            total_bytes: 2,
        });
        assert_eq!(transcript, Transcript::default());
    }
}
