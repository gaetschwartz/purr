//! What `purr sys` says about the engines other than the one whose devices it lists.

use std::path::PathBuf;

use purr_common::model::engine::EngineId;
use purr_core::dev::Device;
use serde::Serialize;

use crate::sys_report;

#[derive(Debug, Clone, Serialize)]
pub struct EngineReport {
    pub engine: EngineId,
    pub status: EngineStatus,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(
    not(feature = "transcribe"),
    allow(dead_code, reason = "only the engines built in construct them")
)]
pub enum EngineStatus {
    Ready {
        version: String,
        directory: Option<PathBuf>,
        devices: Vec<Device>,
    },
    LibraryMissing {
        library: String,
        searched: Vec<PathBuf>,
    },
    LibraryUnloadable {
        library: String,
        directory: PathBuf,
    },
    NoBackend {
        reason: String,
    },
}

impl EngineReport {
    /// The lines of the report, the first one naming the engine and its state.
    pub fn lines(&self) -> Vec<String> {
        let name = self.engine.label();
        match &self.status {
            EngineStatus::Ready {
                version,
                directory,
                devices,
            } => {
                let origin = directory.as_ref().map_or_else(
                    || "linked into the executable".to_owned(),
                    |dir| format!("loaded from {}", dir.display()),
                );
                let mut lines = vec![format!("{name} - Available (version {version}, {origin})")];
                if devices.is_empty() {
                    lines.push("  No devices found.".to_owned());
                }
                for device in devices {
                    lines.push(String::new());
                    lines.extend(
                        sys_report::device_lines(device, &[])
                            .into_iter()
                            .map(|line| format!("  {line}")),
                    );
                }
                lines
            }
            EngineStatus::LibraryMissing { library, searched } => {
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
            EngineStatus::LibraryUnloadable { library, directory } => vec![format!(
                "{name} - Not loadable: {library} in {} cannot be loaded, a library it needs may be missing",
                directory.display()
            )],
            EngineStatus::NoBackend { reason } => {
                vec![format!(
                    "{name} - Not loadable: it registered no compute backend ({reason})"
                )]
            }
        }
    }
}

#[cfg(feature = "transcribe")]
pub fn transcribe() -> EngineReport {
    use purr_core::engine::Engine as _;
    use purr_engine_transcribe::{engine::TranscribeEngine, runtime::Unavailable};

    let engine = TranscribeEngine;
    let status = match engine.availability() {
        Ok(runtime) => EngineStatus::Ready {
            version: runtime.version.clone(),
            directory: runtime.directory.clone(),
            devices: engine.devices(),
        },
        Err(Unavailable::LibraryMissing { library, searched }) => EngineStatus::LibraryMissing {
            library: library.clone(),
            searched: searched.clone(),
        },
        Err(Unavailable::LibraryUnloadable { library, directory }) => {
            EngineStatus::LibraryUnloadable {
                library: library.clone(),
                directory: directory.clone(),
            }
        }
        Err(error @ Unavailable::Backends(_)) => EngineStatus::NoBackend {
            reason: error.to_string(),
        },
    };
    EngineReport {
        engine: engine.id(),
        status,
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    fn report(status: EngineStatus) -> EngineReport {
        EngineReport {
            engine: EngineId::TranscribeCpp,
            status,
        }
    }

    #[test]
    fn a_missing_library_names_it_and_where_it_was_looked_for() {
        let lines = report(EngineStatus::LibraryMissing {
            library: "transcribe.dll".to_owned(),
            searched: vec![PathBuf::from("/a"), PathBuf::from("/b")],
        })
        .lines();
        assert_eq!(
            lines,
            [
                "transcribe.cpp engine - Not loadable: the library transcribe.dll was not found",
                "  looked in /a, /b",
            ]
        );
    }

    #[test]
    fn a_loaded_engine_without_devices_says_so() {
        let lines = report(EngineStatus::Ready {
            version: "0.4.2".to_owned(),
            directory: None,
            devices: Vec::new(),
        })
        .lines();
        assert_eq!(
            lines,
            [
                "transcribe.cpp engine - Available (version 0.4.2, linked into the executable)",
                "  No devices found.",
            ]
        );
    }

    #[test]
    fn a_loaded_engine_lists_each_device_indented() {
        let device = Device {
            id: 0,
            name: "CPU".to_owned(),
            description: String::new(),
            device_id: None,
            vram_free: 0,
            vram_total: 0,
            tpe: purr_common::platform::DeviceType::Cpu,
            backend: Some(purr_core::accel::backend::AccelBackend::Cpu),
            caps: None,
            vulkan_driver: None,
        };
        let lines = report(EngineStatus::Ready {
            version: "0.4.2".to_owned(),
            directory: Some(PathBuf::from("/runtime")),
            devices: vec![device],
        })
        .lines();
        assert_eq!(
            lines.into_iter().take(3).collect::<Vec<_>>(),
            [
                "transcribe.cpp engine - Available (version 0.4.2, loaded from /runtime)",
                "",
                "  Device 0 - CPU",
            ]
        );
    }
}
