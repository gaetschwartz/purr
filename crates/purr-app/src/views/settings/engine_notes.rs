//! What the Hardware page says about the engines that run models: the ones that cannot run, and
//! which engines run on a device.

use purr_common::{
    accel::{EngineAvailability, EngineSummary, EngineUnavailable},
    model::engine::EngineId,
};

fn the(engine: EngineId) -> String {
    format!("The {}", engine.label())
}

/// The sentence about an engine that cannot run, `None` for one that can.
#[must_use]
pub fn engine_note(summary: &EngineSummary) -> Option<String> {
    let engine = the(summary.engine);
    match &summary.availability {
        EngineAvailability::Loaded(_) => None,
        EngineAvailability::NotBuilt => Some(format!(
            "This build of Purr does not include {}.",
            engine.to_lowercase()
        )),
        EngineAvailability::Unavailable(why @ EngineUnavailable::LibraryMissing { .. }) => {
            Some(format!("{engine} is not installed: {why}."))
        }
        EngineAvailability::Unavailable(why @ EngineUnavailable::LibraryUnloadable { .. }) => {
            Some(format!("{engine} cannot be loaded: {why}."))
        }
        EngineAvailability::Unavailable(EngineUnavailable::NoBackend { .. }) => Some(format!(
            "{engine} cannot be loaded: it found no way to compute."
        )),
    }
}

/// One sentence for each engine that cannot run.
#[must_use]
pub fn engine_notes(engines: &[EngineSummary]) -> Vec<String> {
    engines.iter().filter_map(engine_note).collect()
}

fn loaded(engines: &[EngineSummary]) -> usize {
    engines
        .iter()
        .filter(|summary| matches!(summary.availability, EngineAvailability::Loaded(_)))
        .count()
}

/// Which engines run on a device, said only when several engines can run at all.
#[must_use]
pub fn seen_by_note(seen_by: &[EngineId], engines: &[EngineSummary]) -> Option<String> {
    if loaded(engines) < 2 || seen_by.is_empty() {
        return None;
    }
    let names: Vec<String> = seen_by
        .iter()
        .map(|engine| format!("the {}", engine.label()))
        .collect();
    let only = if seen_by.len() < loaded(engines) {
        " only"
    } else {
        ""
    };
    Some(format!("Used by {}{only}", names.join(" and ")))
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

    #[test]
    fn an_engine_that_runs_has_no_note() {
        assert_eq!(engine_note(&loaded_engine(EngineId::WhisperCpp)), None);
    }

    #[test]
    fn a_missing_library_says_the_engine_is_not_installed() {
        assert_eq!(
            engine_note(&unavailable(EngineUnavailable::LibraryMissing {
                library: "transcribe.dll".to_owned(),
                searched: Vec::new(),
            })),
            Some(
                "The transcribe.cpp engine is not installed: transcribe.dll was not found."
                    .to_owned()
            )
        );
    }

    #[test]
    fn a_missing_library_names_the_folders_that_were_searched() {
        assert_eq!(
            engine_note(&unavailable(EngineUnavailable::LibraryMissing {
                library: "transcribe.dll".to_owned(),
                searched: vec![PathBuf::from("runtime"), PathBuf::from("bin")],
            })),
            Some(
                "The transcribe.cpp engine is not installed: transcribe.dll was not found in runtime, bin."
                    .to_owned()
            )
        );
    }

    #[test]
    fn a_library_that_will_not_open_says_where_it_is() {
        assert_eq!(
            engine_note(&unavailable(EngineUnavailable::LibraryUnloadable {
                library: "transcribe.dll".to_owned(),
                directory: PathBuf::from("runtime"),
            })),
            Some(
                "The transcribe.cpp engine cannot be loaded: transcribe.dll in runtime cannot be opened, a library it needs may be missing."
                    .to_owned()
            )
        );
    }

    #[test]
    fn an_engine_without_a_backend_cannot_be_loaded() {
        assert_eq!(
            engine_note(&unavailable(EngineUnavailable::NoBackend {
                detail: String::new()
            })),
            Some(
                "The transcribe.cpp engine cannot be loaded: it found no way to compute."
                    .to_owned()
            )
        );
    }

    #[test]
    fn an_engine_this_build_lacks_is_said_so() {
        assert_eq!(
            engine_note(&EngineSummary {
                engine: EngineId::TranscribeCpp,
                availability: EngineAvailability::NotBuilt,
            }),
            Some("This build of Purr does not include the transcribe.cpp engine.".to_owned())
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
    fn a_device_says_which_engines_use_it_once_several_run() {
        let engines = [
            loaded_engine(EngineId::WhisperCpp),
            loaded_engine(EngineId::TranscribeCpp),
        ];
        assert_eq!(
            seen_by_note(&[EngineId::WhisperCpp, EngineId::TranscribeCpp], &engines),
            Some("Used by the whisper.cpp engine and the transcribe.cpp engine".to_owned())
        );
        assert_eq!(
            seen_by_note(&[EngineId::WhisperCpp], &engines),
            Some("Used by the whisper.cpp engine only".to_owned())
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
        assert_eq!(seen_by_note(&[EngineId::WhisperCpp], &engines), None);
        assert_eq!(
            seen_by_note(
                &[EngineId::WhisperCpp],
                &[loaded_engine(EngineId::WhisperCpp)]
            ),
            None
        );
    }
}
