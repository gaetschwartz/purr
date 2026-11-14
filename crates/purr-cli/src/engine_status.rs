//! What `purr sys` says about each engine: its status lines and its place in the JSON report.

use std::path::{Path, PathBuf};

use purr_common::{
    accel::{EngineAvailability, EngineUnavailable, LoadedEngine},
    model::engine::EngineId,
};
use purr_core::{accel::engine_report::EngineReport, dev::Device};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct EngineJson<'a> {
    engine: EngineId,
    status: StatusJson<'a>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum StatusJson<'a> {
    Ready {
        version: Option<&'a str>,
        directory: Option<&'a Path>,
        devices: &'a [Device],
    },
    LibraryMissing {
        library: &'a str,
        searched: &'a [PathBuf],
    },
    LibraryUnloadable {
        library: &'a str,
        directory: &'a Path,
    },
    NoBackend {
        reason: &'a str,
    },
    NotBuilt,
}

impl<'a> From<&'a EngineReport> for EngineJson<'a> {
    fn from(report: &'a EngineReport) -> Self {
        let status = match &report.availability {
            EngineAvailability::Loaded(loaded) => StatusJson::Ready {
                version: loaded.version.as_deref(),
                directory: loaded.directory.as_deref(),
                devices: &report.inventory.devices,
            },
            EngineAvailability::NotBuilt => StatusJson::NotBuilt,
            EngineAvailability::Unavailable(EngineUnavailable::LibraryMissing {
                library,
                searched,
            }) => StatusJson::LibraryMissing { library, searched },
            EngineAvailability::Unavailable(EngineUnavailable::LibraryUnloadable {
                library,
                directory,
            }) => StatusJson::LibraryUnloadable { library, directory },
            EngineAvailability::Unavailable(EngineUnavailable::NoBackend { detail }) => {
                StatusJson::NoBackend { reason: detail }
            }
        };
        Self {
            engine: report.engine,
            status,
        }
    }
}

fn loaded_line(name: &str, loaded: &LoadedEngine) -> String {
    let origin = loaded.directory.as_ref().map_or_else(
        || "linked into the executable".to_owned(),
        |dir| format!("loaded from {}", dir.display()),
    );
    match &loaded.version {
        Some(version) => format!("{name} - Available (version {version}, {origin})"),
        None => format!("{name} - Available ({origin})"),
    }
}

/// The lines of the report, the first one naming the engine and its state.
pub fn lines(report: &EngineReport) -> Vec<String> {
    let name = report.engine.label();
    match &report.availability {
        EngineAvailability::Loaded(loaded) => {
            let mut lines = vec![loaded_line(name, loaded)];
            if report.inventory.devices.is_empty() {
                lines.push("  No devices found.".to_owned());
            }
            lines
        }
        EngineAvailability::NotBuilt => vec![format!("{name} - Not built into this copy of Purr")],
        EngineAvailability::Unavailable(EngineUnavailable::LibraryMissing {
            library,
            searched,
        }) => {
            let searched = searched
                .iter()
                .map(|dir| dir.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            vec![
                format!("{name} - Not loadable: the library {library} was not found"),
                format!("  looked in {searched}"),
            ]
        }
        EngineAvailability::Unavailable(EngineUnavailable::LibraryUnloadable {
            library,
            directory,
        }) => vec![format!(
            "{name} - Not loadable: {library} in {} cannot be loaded, a library it needs may be missing",
            directory.display()
        )],
        EngineAvailability::Unavailable(EngineUnavailable::NoBackend { detail }) => {
            vec![format!(
                "{name} - Not loadable: it registered no compute backend ({detail})"
            )]
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use purr_core::accel::inventory::Inventory;
    use serde_json::json;

    use super::*;

    fn report(availability: EngineAvailability) -> EngineReport {
        EngineReport::new(EngineId::TranscribeCpp, availability, Inventory::default())
    }

    fn loaded(version: Option<&str>, directory: Option<&str>) -> EngineAvailability {
        EngineAvailability::Loaded(LoadedEngine {
            version: version.map(str::to_owned),
            directory: directory.map(PathBuf::from),
        })
    }

    #[test]
    fn a_missing_library_names_it_and_where_it_was_looked_for() {
        let lines = lines(&report(EngineAvailability::Unavailable(
            EngineUnavailable::LibraryMissing {
                library: "transcribe.dll".to_owned(),
                searched: vec![PathBuf::from("/a"), PathBuf::from("/b")],
            },
        )));
        assert_eq!(
            lines,
            [
                "transcribe.cpp engine - Not loadable: the library transcribe.dll was not found",
                "  looked in /a, /b",
            ]
        );
    }

    #[test]
    fn a_loaded_engine_says_where_it_came_from() {
        assert_eq!(
            lines(&report(loaded(Some("0.4.2"), Some("/runtime"))))
                .into_iter()
                .next(),
            Some(
                "transcribe.cpp engine - Available (version 0.4.2, loaded from /runtime)"
                    .to_owned()
            )
        );
        assert_eq!(
            lines(&report(loaded(None, None))).into_iter().next(),
            Some("transcribe.cpp engine - Available (linked into the executable)".to_owned())
        );
    }

    #[test]
    fn a_loaded_engine_without_devices_says_so() {
        assert_eq!(
            lines(&report(loaded(Some("0.4.2"), None))),
            [
                "transcribe.cpp engine - Available (version 0.4.2, linked into the executable)",
                "  No devices found.",
            ]
        );
    }

    #[test]
    fn an_engine_that_is_not_built_says_so() {
        assert_eq!(
            lines(&report(EngineAvailability::NotBuilt)),
            ["transcribe.cpp engine - Not built into this copy of Purr"]
        );
    }

    #[test]
    fn the_json_keeps_the_status_names_ci_looks_for() {
        let missing = serde_json::to_value(EngineJson::from(&report(
            EngineAvailability::Unavailable(EngineUnavailable::LibraryMissing {
                library: "transcribe.dll".to_owned(),
                searched: Vec::new(),
            }),
        )))
        .unwrap();
        assert_eq!(
            missing,
            json!({
                "engine": "transcribe-cpp",
                "status": { "library_missing": { "library": "transcribe.dll", "searched": [] } }
            })
        );
        let ready =
            serde_json::to_value(EngineJson::from(&report(loaded(Some("0.4.2"), None)))).unwrap();
        assert_eq!(
            ready,
            json!({
                "engine": "transcribe-cpp",
                "status": { "ready": { "version": "0.4.2", "directory": null, "devices": [] } }
            })
        );
    }
}
