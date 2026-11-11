//! The transcript assembled from a stream of transcription updates.

use purr_common::platform::{TranscriptSegment, TranscriptionStatus};
use std::collections::BTreeMap;

/// Segments reported so far for one transcription.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transcript {
    live: BTreeMap<usize, TranscriptSegment>,
    settled: Option<Vec<TranscriptSegment>>,
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
            } => {
                self.live.insert(
                    *chunk_index,
                    TranscriptSegment {
                        text: text.clone(),
                        start_time: *start_time,
                        end_time: *end_time,
                    },
                );
            }
            TranscriptionStatus::Segments { segments } => {
                self.settled = Some(segments.clone());
            }
            TranscriptionStatus::Starting
            | TranscriptionStatus::ProcessingAudio
            | TranscriptionStatus::LoadingModel { .. }
            | TranscriptionStatus::Transcribing
            | TranscriptionStatus::Engine { .. }
            | TranscriptionStatus::LanguageDetected { .. }
            | TranscriptionStatus::Completed { .. }
            | TranscriptionStatus::Error { .. }
            | TranscriptionStatus::InitFailed { .. } => {}
        }
    }

    /// The non-empty segments: the final list once known, otherwise the live ones in order.
    #[must_use]
    pub fn segments(&self) -> Vec<TranscriptSegment> {
        let source: Vec<&TranscriptSegment> = self.settled.as_ref().map_or_else(
            || self.live.values().collect(),
            |settled| settled.iter().collect(),
        );
        source
            .into_iter()
            .filter(|segment| !segment.text.trim().is_empty())
            .cloned()
            .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn in_progress(chunk_index: usize, text: &str, start: f64, end: f64) -> TranscriptionStatus {
        TranscriptionStatus::InProgress {
            chunk_index,
            text: text.to_owned(),
            start_time: start,
            end_time: end,
        }
    }

    fn segment(text: &str, start: f64, end: f64) -> TranscriptSegment {
        TranscriptSegment {
            text: text.to_owned(),
            start_time: start,
            end_time: end,
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
