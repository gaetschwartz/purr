use crate::error::{CliError, CliResult};
use crate::progress::clock;
use purr_core::pipeline::events::EndReason;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

const FAILURE: u8 = 1;
const INTERRUPTED: u8 = 130;

#[derive(Debug)]
pub enum Outcome {
    Done {
        ended: EndReason,
    },
    Skipped,
    Failed {
        error: CliError,
        part: Option<PathBuf>,
    },
    Interrupted {
        part: Option<PathBuf>,
    },
    NotStarted,
}

impl Outcome {
    pub fn into_result(self) -> CliResult<Self> {
        match self {
            Self::Failed { error, .. } => Err(error),
            other @ (Self::Done { .. }
            | Self::Skipped
            | Self::Interrupted { .. }
            | Self::NotStarted) => Ok(other),
        }
    }

    fn part(&self) -> Option<&PathBuf> {
        match self {
            Self::Failed { part, .. } | Self::Interrupted { part } => part.as_ref(),
            Self::Done { .. } | Self::Skipped | Self::NotStarted => None,
        }
    }
}

#[derive(Debug, Default)]
struct Counts {
    done: usize,
    skipped: usize,
    failed: usize,
    interrupted: usize,
    not_started: usize,
}

#[derive(Debug)]
pub struct Entry {
    pub display: String,
    pub outcome: Outcome,
}

#[derive(Debug)]
pub struct Summary {
    entries: Vec<Entry>,
    audio_seconds: f64,
    elapsed: Duration,
}

impl Summary {
    pub const fn new(entries: Vec<Entry>, audio_seconds: f64, elapsed: Duration) -> Self {
        Self {
            entries,
            audio_seconds,
            elapsed,
        }
    }

    pub fn into_result(self) -> CliResult<Self> {
        let Self {
            entries,
            audio_seconds,
            elapsed,
        } = self;
        let entries = entries
            .into_iter()
            .map(|entry| {
                Ok(Entry {
                    display: entry.display,
                    outcome: entry.outcome.into_result()?,
                })
            })
            .collect::<CliResult<Vec<_>>>()?;
        Ok(Self::new(entries, audio_seconds, elapsed))
    }

    fn counts(&self) -> Counts {
        let mut counts = Counts::default();
        for entry in &self.entries {
            match &entry.outcome {
                Outcome::Done { ended } => match ended {
                    EndReason::SourceEnded | EndReason::DurationReached | EndReason::Stopped => {
                        counts.done += 1;
                    }
                },
                Outcome::Skipped => counts.skipped += 1,
                Outcome::Failed { .. } => counts.failed += 1,
                Outcome::Interrupted { .. } => counts.interrupted += 1,
                Outcome::NotStarted => counts.not_started += 1,
            }
        }
        counts
    }

    fn status(&self) -> u8 {
        let counts = self.counts();
        if counts.failed > 0 {
            FAILURE
        } else if counts.interrupted > 0 || counts.not_started > 0 {
            INTERRUPTED
        } else {
            0
        }
    }

    pub fn cut_short(&self) -> bool {
        let counts = self.counts();
        counts.interrupted > 0 || counts.not_started > 0
    }

    pub fn exit_code(&self) -> ExitCode {
        ExitCode::from(self.status())
    }

    pub fn render(&self) -> String {
        let counts = self.counts();
        let tally = [
            (counts.done, "done"),
            (counts.skipped, "skipped"),
            (counts.failed, "failed"),
            (counts.interrupted, "interrupted"),
            (counts.not_started, "not started"),
        ]
        .into_iter()
        .filter(|(count, _)| *count > 0)
        .map(|(count, label)| format!("{count} {label}"))
        .collect::<Vec<_>>()
        .join(", ");

        let mut lines = vec![format!("Summary: {tally}")];

        let failures: Vec<String> = self
            .entries
            .iter()
            .filter_map(|entry| match &entry.outcome {
                Outcome::Failed { error, .. } => Some(format!("  {}: {error}", entry.display)),
                Outcome::Done { .. }
                | Outcome::Skipped
                | Outcome::Interrupted { .. }
                | Outcome::NotStarted => None,
            })
            .collect();
        if !failures.is_empty() {
            lines.push("Failed:".to_owned());
            lines.extend(failures);
        }

        let parts: Vec<String> = self
            .entries
            .iter()
            .filter_map(|entry| entry.outcome.part())
            .map(|part| format!("  {}", part.display()))
            .collect();
        if !parts.is_empty() {
            lines.push("Unfinished output:".to_owned());
            lines.extend(parts);
        }

        let cut: Vec<String> = self
            .entries
            .iter()
            .filter(|entry| match &entry.outcome {
                Outcome::Done { ended } => match ended {
                    EndReason::DurationReached => true,
                    EndReason::SourceEnded | EndReason::Stopped => false,
                },
                Outcome::Skipped
                | Outcome::Failed { .. }
                | Outcome::Interrupted { .. }
                | Outcome::NotStarted => false,
            })
            .map(|entry| format!("  {}", entry.display))
            .collect();
        if !cut.is_empty() {
            lines.push("Ended at the --duration limit:".to_owned());
            lines.extend(cut);
        }

        let audio = self.audio_seconds;
        let elapsed = self.elapsed.as_secs_f64();
        let speed = if elapsed > 0.0 {
            format!(" ({:.1}x real time)", audio / elapsed)
        } else {
            String::new()
        };
        lines.push(format!(
            "Audio: {} in {}{speed}",
            clock(audio),
            clock(elapsed)
        ));
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::path::Path;

    fn entry(display: &str, outcome: Outcome) -> Entry {
        Entry {
            display: display.to_owned(),
            outcome,
        }
    }

    fn done() -> Outcome {
        Outcome::Done {
            ended: EndReason::SourceEnded,
        }
    }

    fn failed(path: &str, part: Option<&str>) -> Outcome {
        Outcome::Failed {
            error: CliError::AudioFileNotFound {
                path: Path::new(path).to_path_buf(),
            },
            part: part.map(PathBuf::from),
        }
    }

    fn status_of(outcomes: Vec<Outcome>) -> u8 {
        let entries = outcomes
            .into_iter()
            .map(|outcome| entry("x", outcome))
            .collect();
        Summary::new(entries, 0.0, Duration::ZERO).status()
    }

    #[test]
    fn a_run_without_incident_exits_zero() {
        assert_eq!(status_of(vec![done(), done()]), 0);
        assert_eq!(status_of(vec![Outcome::Skipped]), 0);
        assert_eq!(status_of(vec![done(), Outcome::Skipped]), 0);
    }

    #[test]
    fn stopping_a_source_of_unknown_length_is_a_normal_end() {
        let stopped = Outcome::Done {
            ended: EndReason::Stopped,
        };
        assert_eq!(status_of(vec![stopped]), 0);
    }

    #[test]
    fn any_failure_exits_one_even_beside_interrupted_inputs() {
        assert_eq!(status_of(vec![done(), failed("a.wav", None)]), 1);
        assert_eq!(
            status_of(vec![
                failed("a.wav", None),
                Outcome::Interrupted { part: None },
                Outcome::NotStarted
            ]),
            1
        );
    }

    #[test]
    fn interrupted_or_unstarted_inputs_without_failures_exit_130() {
        assert_eq!(
            status_of(vec![done(), Outcome::Interrupted { part: None }]),
            130
        );
        assert_eq!(status_of(vec![done(), Outcome::NotStarted]), 130);
    }

    #[test]
    fn the_summary_lists_counts_failures_and_unfinished_output() {
        let entries = vec![
            entry("a.wav", done()),
            entry("b.wav", Outcome::Skipped),
            entry("missing.wav", failed("missing.wav", None)),
            entry(
                "broken.wav",
                failed("broken.wav", Some("out\\broken.txt.part")),
            ),
            entry(
                "c.wav",
                Outcome::Interrupted {
                    part: Some(PathBuf::from("out\\c.txt.part")),
                },
            ),
            entry("d.wav", Outcome::NotStarted),
        ];
        let summary = Summary::new(entries, 125.0, Duration::from_secs(25));
        assert_eq!(
            summary.render(),
            "Summary: 1 done, 1 skipped, 2 failed, 1 interrupted, 1 not started\n\
             Failed:\n  \
             missing.wav: Media file not found: missing.wav\n  \
             broken.wav: Media file not found: broken.wav\n\
             Unfinished output:\n  \
             out\\broken.txt.part\n  \
             out\\c.txt.part\n\
             Audio: 2:05 in 0:25 (5.0x real time)"
        );
    }

    #[test]
    fn inputs_cut_at_the_duration_are_named() {
        let entries = vec![
            entry("a.wav", done()),
            entry(
                "radio",
                Outcome::Done {
                    ended: EndReason::DurationReached,
                },
            ),
            entry(
                "other",
                Outcome::Done {
                    ended: EndReason::Stopped,
                },
            ),
        ];
        let summary = Summary::new(entries, 60.0, Duration::from_secs(30));
        assert_eq!(
            summary.render(),
            "Summary: 3 done\n\
             Ended at the --duration limit:\n  \
             radio\n\
             Audio: 1:00 in 0:30 (2.0x real time)"
        );
    }

    #[test]
    fn only_interrupted_or_unstarted_inputs_cut_a_run_short() {
        let summary = |outcomes: Vec<Outcome>| {
            Summary::new(
                outcomes.into_iter().map(|o| entry("x", o)).collect(),
                0.0,
                Duration::ZERO,
            )
            .cut_short()
        };
        assert!(!summary(vec![done(), failed("a.wav", None)]));
        assert!(summary(vec![done(), Outcome::Interrupted { part: None }]));
        assert!(summary(vec![done(), Outcome::NotStarted]));
    }

    #[test]
    fn a_clean_run_has_no_failure_or_output_sections() {
        let entries = vec![entry("a.wav", done()), entry("b.wav", done())];
        let summary = Summary::new(entries, 60.0, Duration::from_secs(30));
        assert_eq!(
            summary.render(),
            "Summary: 2 done\nAudio: 1:00 in 0:30 (2.0x real time)"
        );
    }

    #[test]
    fn a_failed_outcome_becomes_its_error() {
        assert!(matches!(
            failed("a.wav", None).into_result(),
            Err(CliError::AudioFileNotFound { .. })
        ));
        assert!(matches!(done().into_result(), Ok(Outcome::Done { .. })));
    }
}
