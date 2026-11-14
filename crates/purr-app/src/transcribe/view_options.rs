//! Which of the "Details" options of the More menu a transcript can offer, and why not.

use purr_common::model::{descriptor::ModelDescriptor, engine::EngineId};

/// Where the transcript stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Moment {
    Running,
    /// The run stopped with an error
    Stopped,
    Finished,
    /// A transcription opened from History
    Saved,
}

/// What the transcript and its audio hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Evidence {
    /// Some word carries a score
    pub scores: bool,
    /// Some word carries a time
    pub times: bool,
    /// The audio can be played
    pub playable: bool,
}

/// What one option of the menu shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Offer {
    Switch,
    /// A line saying why the option is not there
    Reason(&'static str),
}

/// The "Details" rows of the More menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Details {
    /// The engine of the model reports none of it: one line stands for all
    Unreported(EngineId),
    Desktop {
        unsure: Offer,
        word_seek: Offer,
    },
}

/// Why a model that reports no word scores or word times has no rows.
#[must_use]
pub fn unreported_reason(engine: EngineId) -> String {
    format!(
        "The {} reports no word scores or word times.",
        engine.label()
    )
}

const NOT_RECORDED: &str = "Not recorded for this transcription.";

/// The rows to show for a transcript at `moment`, made by `model` when it is known; a
/// transcript whose model is not known is judged by its `evidence`.
#[must_use]
pub fn details(model: Option<&ModelDescriptor>, moment: Moment, evidence: Evidence) -> Details {
    if let Some(model) = model.filter(|model| !model.profile.reports_word_detail()) {
        return Details::Unreported(model.engine);
    }
    let unsure = match moment {
        Moment::Running => Offer::Switch,
        Moment::Stopped | Moment::Finished | Moment::Saved => {
            if evidence.scores {
                Offer::Switch
            } else {
                Offer::Reason(NOT_RECORDED)
            }
        }
    };
    let word_seek = match moment {
        Moment::Running => Offer::Reason("Available when the transcription ends."),
        Moment::Stopped => Offer::Reason("Not available after a stopped transcription."),
        Moment::Finished | Moment::Saved => {
            if !evidence.playable {
                Offer::Reason("The audio isn't available.")
            } else if !evidence.times {
                Offer::Reason(NOT_RECORDED)
            } else {
                Offer::Switch
            }
        }
    };
    Details::Desktop { unsure, word_seek }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_catalog::test_support::{descriptor, weights};

    const FULL: Evidence = Evidence {
        scores: true,
        times: true,
        playable: true,
    };

    fn whisper() -> ModelDescriptor {
        descriptor("tiny", weights(Some("q5_1")), false)
    }

    fn browser() -> ModelDescriptor {
        descriptor("tiny", weights(Some("fp32/q4")), false)
    }

    #[test]
    fn the_browser_gets_one_line_whatever_the_moment() {
        for moment in [
            Moment::Running,
            Moment::Stopped,
            Moment::Finished,
            Moment::Saved,
        ] {
            assert_eq!(
                details(Some(&browser()), moment, FULL),
                Details::Unreported(EngineId::Browser)
            );
        }
    }

    #[test]
    fn the_line_names_what_the_engine_lacks() {
        assert_eq!(
            unreported_reason(EngineId::Browser),
            "The browser engine reports no word scores or word times."
        );
    }

    #[test]
    fn a_model_that_reports_words_gets_the_rows_whatever_the_engine() {
        assert_eq!(
            details(Some(&whisper()), Moment::Finished, FULL),
            details(None, Moment::Finished, FULL)
        );
    }

    #[test]
    fn a_model_that_times_words_but_scores_none_still_gets_the_rows() {
        let mut model = browser();
        model.profile.timing = purr_common::model::profile::Timing::Word;
        assert_eq!(
            details(Some(&model), Moment::Finished, FULL),
            Details::Desktop {
                unsure: Offer::Switch,
                word_seek: Offer::Switch,
            }
        );
    }

    #[test]
    fn a_running_transcript_offers_marks_but_not_playback() {
        assert_eq!(
            details(
                Some(&whisper()),
                Moment::Running,
                Evidence {
                    scores: false,
                    times: false,
                    playable: true
                }
            ),
            Details::Desktop {
                unsure: Offer::Switch,
                word_seek: Offer::Reason("Available when the transcription ends."),
            }
        );
    }

    #[test]
    fn a_finished_transcript_offers_marks_and_playback() {
        assert_eq!(
            details(Some(&whisper()), Moment::Finished, FULL),
            Details::Desktop {
                unsure: Offer::Switch,
                word_seek: Offer::Switch,
            }
        );
    }

    #[test]
    fn what_was_not_recorded_or_cannot_play_says_so() {
        assert_eq!(
            details(
                Some(&whisper()),
                Moment::Saved,
                Evidence {
                    scores: false,
                    times: false,
                    playable: true
                }
            ),
            Details::Desktop {
                unsure: Offer::Reason("Not recorded for this transcription."),
                word_seek: Offer::Reason("Not recorded for this transcription."),
            }
        );
        assert_eq!(
            details(
                Some(&whisper()),
                Moment::Saved,
                Evidence {
                    playable: false,
                    ..FULL
                }
            ),
            Details::Desktop {
                unsure: Offer::Switch,
                word_seek: Offer::Reason("The audio isn't available."),
            }
        );
    }

    #[test]
    fn a_stopped_run_cannot_play_from_a_word() {
        assert_eq!(
            details(Some(&whisper()), Moment::Stopped, FULL),
            Details::Desktop {
                unsure: Offer::Switch,
                word_seek: Offer::Reason("Not available after a stopped transcription."),
            }
        );
    }
}
