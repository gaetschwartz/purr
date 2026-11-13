//! The read-only system panel as plain rows, made from what the platform reports.

use crate::hooks::use_models::GpuSupport;
use purr_common::accel::{AccelBackend, BackendStatus, BackendUnavailable, SystemReport};

/// The status badge at the end of a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Badge {
    Available,
    BuiltUnavailable,
    Unavailable,
}

impl Badge {
    /// The words on the badge.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Available => "Available",
            Self::BuiltUnavailable | Self::Unavailable => "Unavailable",
        }
    }
}

/// One line of the panel: a backend, the device it uses or why it cannot run, and its badge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemRow {
    pub backend: &'static str,
    pub name: String,
    pub detail: Option<String>,
    pub badge: Badge,
}

fn missing_library_detail(searched: &[String]) -> String {
    if searched.is_empty() {
        "Purr supports it, but the library it needs is not installed.".to_owned()
    } else {
        format!(
            "Purr supports it, but the library it needs is not installed. Looked in {}, then the system's library folders.",
            searched.join(", ")
        )
    }
}

fn unavailable_row(backend: AccelBackend, why: &BackendUnavailable) -> Option<SystemRow> {
    let (name, detail) = match why {
        BackendUnavailable::NotBuilt | BackendUnavailable::NotBuiltOrNoDevice => return None,
        BackendUnavailable::NoDevice => (
            "No usable device found".to_owned(),
            Some("Purr supports it, but no device or driver responded.".to_owned()),
        ),
        BackendUnavailable::RuntimeLibraryMissing { library, searched } => (
            format!("Missing {library}"),
            Some(missing_library_detail(searched)),
        ),
    };
    Some(SystemRow {
        backend: backend.label(),
        name,
        detail,
        badge: Badge::BuiltUnavailable,
    })
}

/// Rows for the backends this build includes that cannot run here, with why. They have no device
/// to list, so the device list cannot show them.
#[must_use]
pub fn unavailable_rows(report: &SystemReport) -> Vec<SystemRow> {
    report
        .backends
        .iter()
        .filter_map(|entry| match &entry.status {
            BackendStatus::Available => None,
            BackendStatus::Unavailable(why) => unavailable_row(entry.backend, why),
        })
        .collect()
}

/// The rows for the browser's WebGPU adapter.
#[must_use]
pub fn browser_rows(gpu: &GpuSupport) -> Vec<SystemRow> {
    let (name, detail, badge) = match gpu {
        GpuSupport::Checking => ("Checking this browser…".to_owned(), None, Badge::Unavailable),
        GpuSupport::Unavailable => (
            "No WebGPU adapter found".to_owned(),
            Some(
                "Purr uses the CPU here, which is slower. Use a current Chrome, Edge or another browser with WebGPU enabled."
                    .to_owned(),
            ),
            Badge::Unavailable,
        ),
        GpuSupport::Available { name, shader_f16 } => (
            name.clone()
                .unwrap_or_else(|| "WebGPU adapter".to_owned()),
            Some(if *shader_f16 {
                "16-bit floats supported".to_owned()
            } else {
                "No 16-bit floats: the Large v3 Turbo model is unavailable".to_owned()
            }),
            Badge::Available,
        ),
    };
    vec![SystemRow {
        backend: "WebGPU",
        name,
        detail,
        badge,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::accel::{BackendReport, CpuReport};

    fn backend(backend: AccelBackend, status: BackendStatus) -> purr_common::accel::BackendReport {
        BackendReport {
            backend,
            status,
            devices: vec![],
        }
    }

    fn row(backend: &'static str, name: &str, detail: Option<&str>, badge: Badge) -> SystemRow {
        SystemRow {
            backend,
            name: name.to_owned(),
            detail: detail.map(str::to_owned),
            badge,
        }
    }

    #[test]
    fn only_backends_this_build_has_but_cannot_run_get_a_row() {
        let report = SystemReport {
            backends: vec![
                backend(
                    AccelBackend::Cuda,
                    BackendStatus::Unavailable(BackendUnavailable::RuntimeLibraryMissing {
                        library: "cudart64_13.dll".to_owned(),
                        searched: vec!["runtime".to_owned()],
                    }),
                ),
                backend(
                    AccelBackend::Metal,
                    BackendStatus::Unavailable(BackendUnavailable::NotBuilt),
                ),
                backend(
                    AccelBackend::Hip,
                    BackendStatus::Unavailable(BackendUnavailable::NoDevice),
                ),
                backend(AccelBackend::Cpu, BackendStatus::Available),
            ],
            cpu: CpuReport {
                name: None,
                logical_threads: None,
                built_with: vec![],
                detected: None,
            },
            last_run: None,
        };
        assert_eq!(
            unavailable_rows(&report),
            [
                row(
                    "CUDA",
                    "Missing cudart64_13.dll",
                    Some(
                        "Purr supports it, but the library it needs is not installed. Looked in runtime, then the system's library folders."
                    ),
                    Badge::BuiltUnavailable
                ),
                row(
                    "ROCm",
                    "No usable device found",
                    Some("Purr supports it, but no device or driver responded."),
                    Badge::BuiltUnavailable
                ),
            ]
        );
    }

    #[test]
    fn the_browser_reports_its_adapter_and_what_it_lacks() {
        let with = |shader_f16| {
            browser_rows(&GpuSupport::Available {
                name: Some("Some Adapter".to_owned()),
                shader_f16,
            })
        };
        assert_eq!(
            with(true),
            [row(
                "WebGPU",
                "Some Adapter",
                Some("16-bit floats supported"),
                Badge::Available
            )]
        );
        assert_eq!(
            with(false),
            [row(
                "WebGPU",
                "Some Adapter",
                Some("No 16-bit floats: the Large v3 Turbo model is unavailable"),
                Badge::Available
            )]
        );
    }

    #[test]
    fn a_browser_without_webgpu_cannot_transcribe() {
        let rows = browser_rows(&GpuSupport::Unavailable);
        assert_eq!(
            rows.iter().map(|row| row.badge).collect::<Vec<_>>(),
            [Badge::Unavailable]
        );
        assert_eq!(
            rows.first().map(|row| row.name.as_str()),
            Some("No WebGPU adapter found")
        );
    }
}
