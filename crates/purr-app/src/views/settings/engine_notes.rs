//! What the Hardware page says about the engines that run models: the ones that cannot run, and
//! the devices only some engines use.

use purr_common::{
    accel::{EngineAvailability, EngineSummary, EngineUnavailable},
    model::engine::EngineId,
};

/// Why an engine cannot run, in the headline of its note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineProblem {
    NotBuilt,
    NotInstalled,
    CannotLoad,
}

impl EngineProblem {
    /// The words after the engine's name.
    #[must_use]
    pub const fn words(self) -> &'static str {
        match self {
            Self::NotBuilt => "not in this build",
            Self::NotInstalled => "not installed",
            Self::CannotLoad => "can't be loaded",
        }
    }
}

/// An engine that cannot run: what is wrong, and the cause behind a disclosure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineNote {
    pub engine: EngineId,
    pub problem: EngineProblem,
    pub detail: Option<String>,
}

impl EngineNote {
    #[must_use]
    pub fn headline(&self) -> String {
        format!("{} {}", self.engine.label(), self.problem.words())
    }
}

/// The note about an engine that cannot run, `None` for one that can.
#[must_use]
pub fn engine_note(summary: &EngineSummary) -> Option<EngineNote> {
    let (problem, detail) = match &summary.availability {
        EngineAvailability::Loaded(_) => return None,
        EngineAvailability::NotBuilt => (EngineProblem::NotBuilt, None),
        EngineAvailability::Unavailable(why @ EngineUnavailable::LibraryMissing { .. }) => {
            (EngineProblem::NotInstalled, Some(why.to_string()))
        }
        EngineAvailability::Unavailable(why @ EngineUnavailable::LibraryUnloadable { .. }) => {
            (EngineProblem::CannotLoad, Some(why.to_string()))
        }
        EngineAvailability::Unavailable(EngineUnavailable::NoBackend { detail }) => (
            EngineProblem::CannotLoad,
            (!detail.trim().is_empty()).then(|| detail.clone()),
        ),
    };
    Some(EngineNote {
        engine: summary.engine,
        problem,
        detail,
    })
}

/// One note for each engine that cannot run.
#[must_use]
pub fn engine_notes(engines: &[EngineSummary]) -> Vec<EngineNote> {
    engines.iter().filter_map(engine_note).collect()
}

fn loaded(engines: &[EngineSummary]) -> usize {
    engines
        .iter()
        .filter(|summary| matches!(summary.availability, EngineAvailability::Loaded(_)))
        .count()
}

/// The engines that use a device, when some engine that runs does not.
#[must_use]
pub fn only_used_by(seen_by: &[EngineId], engines: &[EngineSummary]) -> Option<Vec<EngineId>> {
    (loaded(engines) >= 2 && !seen_by.is_empty() && seen_by.len() < loaded(engines))
        .then(|| seen_by.to_vec())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use purr_common::accel::LoadedEngine;

    use super::*;

    fn loaded_engine(engine: EngineId) -> EngineSummary {
        EngineSummary {
            engine,
            availability: EngineAvailability::Loaded(LoadedEngine {
                version: None,
                directory: None,
            }),
        }
    }

    fn unavailable(why: EngineUnavailable) -> EngineSummary {
        EngineSummary {
            engine: EngineId::TranscribeCpp,
            availability: EngineAvailability::Unavailable(why),
        }
    }

    fn note(problem: EngineProblem, detail: Option<String>) -> Option<EngineNote> {
        Some(EngineNote {
            engine: EngineId::TranscribeCpp,
            problem,
            detail,
        })
    }

    #[test]
    fn an_engine_that_runs_has_no_note() {
        assert_eq!(engine_note(&loaded_engine(EngineId::WhisperCpp)), None);
    }

    #[test]
    fn a_missing_library_keeps_the_error_as_the_detail() {
        let why = EngineUnavailable::LibraryMissing {
            library: "transcribe.dll".to_owned(),
            searched: vec![PathBuf::from("runtime"), PathBuf::from("bin")],
        };
        assert_eq!(
            engine_note(&unavailable(why.clone())),
            note(EngineProblem::NotInstalled, Some(why.to_string()))
        );
    }

    #[test]
    fn a_library_that_will_not_open_cannot_be_loaded() {
        let why = EngineUnavailable::LibraryUnloadable {
            library: "transcribe.dll".to_owned(),
            directory: PathBuf::from("runtime"),
        };
        assert_eq!(
            engine_note(&unavailable(why.clone())),
            note(EngineProblem::CannotLoad, Some(why.to_string()))
        );
    }

    #[test]
    fn an_engine_without_a_backend_cannot_be_loaded() {
        assert_eq!(
            engine_note(&unavailable(EngineUnavailable::NoBackend {
                detail: String::new()
            })),
            note(EngineProblem::CannotLoad, None)
        );
        assert_eq!(
            engine_note(&unavailable(EngineUnavailable::NoBackend {
                detail: "no backend registered".to_owned()
            })),
            note(
                EngineProblem::CannotLoad,
                Some("no backend registered".to_owned())
            )
        );
    }

    #[test]
    fn an_engine_this_build_lacks_has_no_detail() {
        assert_eq!(
            engine_note(&EngineSummary {
                engine: EngineId::TranscribeCpp,
                availability: EngineAvailability::NotBuilt,
            }),
            note(EngineProblem::NotBuilt, None)
        );
    }

    #[test]
    fn only_the_engines_that_cannot_run_are_listed() {
        let engines = [
            loaded_engine(EngineId::WhisperCpp),
            unavailable(EngineUnavailable::NoBackend {
                detail: String::new(),
            }),
        ];
        assert_eq!(engine_notes(&engines).len(), 1);
    }

    #[test]
    fn a_device_names_its_engines_only_when_another_running_engine_skips_it() {
        let engines = [
            loaded_engine(EngineId::WhisperCpp),
            loaded_engine(EngineId::TranscribeCpp),
        ];
        assert_eq!(
            only_used_by(&[EngineId::WhisperCpp, EngineId::TranscribeCpp], &engines),
            None
        );
        assert_eq!(
            only_used_by(&[EngineId::WhisperCpp], &engines),
            Some(vec![EngineId::WhisperCpp])
        );
    }

    #[test]
    fn with_one_engine_running_a_device_says_nothing_about_engines() {
        let engines = [
            loaded_engine(EngineId::WhisperCpp),
            unavailable(EngineUnavailable::NoBackend {
                detail: String::new(),
            }),
        ];
        assert_eq!(only_used_by(&[EngineId::WhisperCpp], &engines), None);
        assert_eq!(
            only_used_by(
                &[EngineId::WhisperCpp],
                &[loaded_engine(EngineId::WhisperCpp)]
            ),
            None
        );
    }
}
