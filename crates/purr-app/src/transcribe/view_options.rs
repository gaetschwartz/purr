//! Which of the "Details" options of the More menu a transcript can offer, and why not.

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
    Absent,
}

/// The "Details" rows of the More menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Details {
    /// The browser engine reports none of it: one line stands for all
    Browser,
    Desktop {
        unsure: Offer,
        word_seek: Offer,
        beams: Offer,
    },
}

pub const BROWSER_REASON: &str = "The browser engine reports no word scores, word times or beams.";
const NOT_RECORDED: &str = "Not recorded for this transcription.";

/// The rows to show for a transcript at `moment`.
#[must_use]
pub const fn details(web: bool, moment: Moment, evidence: Evidence) -> Details {
    if web {
        return Details::Browser;
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
    let beams = match moment {
        Moment::Running => Offer::Switch,
        Moment::Stopped | Moment::Finished => Offer::Reason("Shown only while transcribing."),
        Moment::Saved => Offer::Absent,
    };
    Details::Desktop {
        unsure,
        word_seek,
        beams,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: Evidence = Evidence {
        scores: true,
        times: true,
        playable: true,
    };

    #[test]
    fn the_browser_gets_one_line_whatever_the_moment() {
        for moment in [
            Moment::Running,
            Moment::Stopped,
            Moment::Finished,
            Moment::Saved,
        ] {
            assert_eq!(details(true, moment, FULL), Details::Browser);
        }
    }

    #[test]
    fn a_running_transcript_offers_marks_and_beams_but_not_playback() {
        assert_eq!(
            details(
                false,
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
                beams: Offer::Switch,
            }
        );
    }

    #[test]
    fn a_finished_transcript_offers_marks_and_playback_but_not_beams() {
        assert_eq!(
            details(false, Moment::Finished, FULL),
            Details::Desktop {
                unsure: Offer::Switch,
                word_seek: Offer::Switch,
                beams: Offer::Reason("Shown only while transcribing."),
            }
        );
    }

    #[test]
    fn a_saved_transcript_has_no_beams_row() {
        assert_eq!(
            details(false, Moment::Saved, FULL),
            Details::Desktop {
                unsure: Offer::Switch,
                word_seek: Offer::Switch,
                beams: Offer::Absent,
            }
        );
    }

    #[test]
    fn what_was_not_recorded_or_cannot_play_says_so() {
        assert_eq!(
            details(
                false,
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
                beams: Offer::Absent,
            }
        );
        assert_eq!(
            details(
                false,
                Moment::Saved,
                Evidence {
                    playable: false,
                    ..FULL
                }
            ),
            Details::Desktop {
                unsure: Offer::Switch,
                word_seek: Offer::Reason("The audio isn't available."),
                beams: Offer::Absent,
            }
        );
    }

    #[test]
    fn a_stopped_run_cannot_play_from_a_word() {
        assert_eq!(
            details(false, Moment::Stopped, FULL),
            Details::Desktop {
                unsure: Offer::Switch,
                word_seek: Offer::Reason("Not available after a stopped transcription."),
                beams: Offer::Reason("Shown only while transcribing."),
            }
        );
    }
}
