//! `purr engines`: the report is the command's result, on stdout in every mode.

use miette::IntoDiagnostic as _;
use purr_common::accel::{CpuFeature, EngineAvailability};
use purr_core::accel::{
    backend::{AccelBackend, BuildStatus},
    build_info::BuildInfo,
    engine_report::{self, EngineReport},
};
use serde::Serialize;

use crate::{engine_status, engine_status::EngineJson, engines, resultln, sys_report};

#[derive(Debug, Serialize)]
struct EnginesJson<'a> {
    engines: Vec<EngineEntry<'a>>,
}

#[derive(Debug, Serialize)]
struct EngineEntry<'a> {
    #[serde(flatten)]
    engine: EngineJson<'a>,
    built: bool,
    available: bool,
    build: BuildJson,
}

#[derive(Debug, Serialize)]
struct BuildJson {
    backends: Vec<BackendBuildJson>,
    cpu: Vec<CpuFeature>,
}

#[derive(Debug, Serialize)]
struct BackendBuildJson {
    backend: AccelBackend,
    status: BuildStatus,
}

impl<'a> From<&'a EngineReport> for EngineEntry<'a> {
    fn from(report: &'a EngineReport) -> Self {
        Self {
            engine: EngineJson::from(report),
            built: !matches!(report.availability, EngineAvailability::NotBuilt),
            available: report.is_loaded(),
            build: BuildJson::from(&report.inventory.build),
        }
    }
}

impl From<&BuildInfo> for BuildJson {
    fn from(build: &BuildInfo) -> Self {
        let backends = if build.backends.is_empty() {
            Vec::new()
        } else {
            AccelBackend::DEFAULT_PRIORITY
                .iter()
                .map(|backend| BackendBuildJson {
                    backend: *backend,
                    status: build.status(*backend),
                })
                .collect()
        };
        Self {
            backends,
            cpu: build.cpu.clone(),
        }
    }
}

fn json(reports: &[EngineReport]) -> serde_json::Result<String> {
    serde_json::to_string_pretty(&EnginesJson {
        engines: reports.iter().map(EngineEntry::from).collect(),
    })
}

fn build_lines(build: &BuildInfo) -> Vec<String> {
    let mut lines = Vec::new();
    if !build.backends.is_empty() {
        let built: Vec<&str> = AccelBackend::DEFAULT_PRIORITY
            .iter()
            .filter(|backend| build.status(**backend) == BuildStatus::Built)
            .map(|backend| backend.pretty_name())
            .collect();
        let names = if built.is_empty() {
            "none".to_owned()
        } else {
            built.join(", ")
        };
        lines.push(format!("  Backends: {names}"));
    }
    if !build.cpu.is_empty() {
        lines.push(format!(
            "  CPU instruction sets: {}",
            CpuFeature::join_labels(&build.cpu)
        ));
    }
    lines
}

fn engine_lines(report: &EngineReport) -> Vec<String> {
    let mut lines = engine_status::lines(report);
    lines.extend(build_lines(&report.inventory.build));
    for device in &report.inventory.devices {
        lines.extend(
            sys_report::device_lines(device, &[])
                .into_iter()
                .map(|line| format!("  {line}")),
        );
    }
    lines
}

fn lines(reports: &[EngineReport]) -> Vec<String> {
    let mut lines = Vec::new();
    for (position, report) in reports.iter().enumerate() {
        if position > 0 {
            lines.push(String::new());
        }
        lines.extend(engine_lines(report));
    }
    lines
}

pub fn handle(json_output: bool) -> miette::Result<()> {
    let reports = engine_report::reports(&engines::all(), &engines::KNOWN);
    if json_output {
        resultln!("{}", json(&reports).into_diagnostic()?)?;
        return Ok(());
    }
    for line in lines(&reports) {
        resultln!("{line}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use pretty_assertions::assert_eq;
    use purr_common::{
        accel::{EngineUnavailable, LoadedEngine},
        model::engine::EngineId,
        platform::DeviceType,
    };
    use purr_core::{accel::inventory::Inventory, dev::Device};
    use serde_json::json;

    use super::*;

    fn build() -> BuildInfo {
        BuildInfo {
            backends: HashMap::from([
                (AccelBackend::Cpu, BuildStatus::Built),
                (AccelBackend::Vulkan, BuildStatus::Built),
                (AccelBackend::Cuda, BuildStatus::NotBuilt),
            ]),
            cpu: vec![CpuFeature::Avx2, CpuFeature::Fma],
        }
    }

    fn processor() -> Device {
        Device {
            id: 0,
            name: "CPU".to_owned(),
            description: "AMD Ryzen 9 7950X 16-Core Processor".to_owned(),
            device_id: None,
            vram_free: 0,
            vram_total: 0,
            tpe: DeviceType::Cpu,
            backend: Some(AccelBackend::Cpu),
            caps: None,
            vulkan_driver: None,
        }
    }

    fn loaded() -> EngineReport {
        EngineReport::new(
            EngineId::WhisperCpp,
            EngineAvailability::Loaded(LoadedEngine {
                version: None,
                directory: None,
            }),
            Inventory {
                devices: vec![processor()],
                build: build(),
                ..Inventory::default()
            },
        )
    }

    fn library_missing() -> EngineReport {
        EngineReport::new(
            EngineId::TranscribeCpp,
            EngineAvailability::Unavailable(EngineUnavailable::LibraryMissing {
                library: "transcribe.dll".to_owned(),
                searched: vec![PathBuf::from("/a")],
            }),
            Inventory {
                build: build(),
                ..Inventory::default()
            },
        )
    }

    #[test]
    fn a_loaded_engine_lists_its_build_and_its_devices() {
        assert_eq!(
            engine_lines(&loaded()),
            [
                "whisper.cpp engine - Available (linked into the executable)",
                "  Backends: Vulkan, CPU",
                "  CPU instruction sets: AVX2, FMA",
                "  Device 0 - CPU",
                "    Description: AMD Ryzen 9 7950X 16-Core Processor",
                "    Type: CPU",
                "    Backend: CPU",
                "    Device id: not reported by ggml",
                "    Memory: not reported",
                "    Capabilities: not reported",
            ]
        );
    }

    #[test]
    fn an_engine_that_cannot_load_gives_its_reason_and_its_build_but_no_devices() {
        assert_eq!(
            engine_lines(&library_missing()),
            [
                "transcribe.cpp engine - Not loadable: the library transcribe.dll was not found",
                "  looked in /a",
                "  Backends: Vulkan, CPU",
                "  CPU instruction sets: AVX2, FMA",
            ]
        );
    }

    #[test]
    fn an_engine_that_is_not_built_is_one_line() {
        assert_eq!(
            engine_lines(&EngineReport::not_built(EngineId::TranscribeCpp)),
            ["transcribe.cpp engine - Not built into this copy of Purr"]
        );
    }

    #[test]
    fn a_build_without_a_backend_says_none() {
        let build = BuildInfo {
            backends: HashMap::from([(AccelBackend::Cuda, BuildStatus::NotBuilt)]),
            cpu: Vec::new(),
        };
        assert_eq!(build_lines(&build), ["  Backends: none"]);
    }

    #[test]
    fn engines_are_separated_by_a_blank_line() {
        let reports = [loaded(), EngineReport::not_built(EngineId::TranscribeCpp)];
        let all = lines(&reports);
        let (first, second) = all.split_at(all.len() - 2);
        assert_eq!(
            first.last().map(String::as_str),
            Some("    Capabilities: not reported")
        );
        assert_eq!(
            second,
            [
                "",
                "transcribe.cpp engine - Not built into this copy of Purr"
            ]
        );
    }

    #[test]
    fn the_json_says_whether_each_engine_is_built_and_available() {
        let reports = [
            library_missing(),
            EngineReport::not_built(EngineId::TranscribeCpp),
        ];
        let document: serde_json::Value = serde_json::from_str(&json(&reports).unwrap()).unwrap();
        let backends = json!([
            { "backend": "cuda", "status": "not_built" },
            { "backend": "hip", "status": "unknown" },
            { "backend": "coreml", "status": "unknown" },
            { "backend": "metal", "status": "unknown" },
            { "backend": "sycl", "status": "unknown" },
            { "backend": "vulkan", "status": "built" },
            { "backend": "cpu", "status": "built" },
        ]);
        assert_eq!(
            document,
            json!({
                "engines": [
                    {
                        "engine": "transcribe-cpp",
                        "status": {
                            "library_missing": { "library": "transcribe.dll", "searched": ["/a"] }
                        },
                        "built": true,
                        "available": false,
                        "build": { "backends": backends, "cpu": ["Avx2", "Fma"] }
                    },
                    {
                        "engine": "transcribe-cpp",
                        "status": "not_built",
                        "built": false,
                        "available": false,
                        "build": { "backends": [], "cpu": [] }
                    }
                ]
            })
        );
    }

    #[test]
    fn a_loaded_engine_is_available_in_the_json() {
        let report = loaded();
        let entry = serde_json::to_value(EngineEntry::from(&report)).unwrap();
        assert_eq!(entry["built"], json!(true));
        assert_eq!(entry["available"], json!(true));
        assert_eq!(entry["status"]["ready"]["devices"][0]["name"], json!("CPU"));
    }
}
