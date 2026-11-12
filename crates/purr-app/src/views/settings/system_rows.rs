//! The read-only system panel as plain rows, made from what the platform reports.

use crate::{
    hooks::use_models::GpuSupport,
    transcribe::engine::{routinely_skipped, why_passed_over},
    utils::byte_size::ByteSize,
};
use purr_common::{
    accel::{
        AccelBackend, AccelSelection, BackendReport, BackendStatus, BackendUnavailable, CpuFeature,
        CpuReport, LastRun, SystemReport,
    },
    platform::DeviceInfo,
};

/// The status badge at the end of a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Badge {
    Available,
    BuiltUnavailable,
    NotBuilt,
    Unavailable,
}

impl Badge {
    /// The words on the badge.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Available => "Available",
            Self::BuiltUnavailable => "Built, unavailable",
            Self::NotBuilt => "Not built",
            Self::Unavailable => "Not available",
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

fn device_name(device: &DeviceInfo) -> String {
    match device.description.as_deref().map(str::trim) {
        Some(description) if !description.is_empty() => description.to_owned(),
        Some(_) | None => device.name.clone(),
    }
}

fn cpu_detail(cpu: &CpuReport) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(threads) = cpu.logical_threads {
        parts.push(format!("{threads} threads"));
    }
    if !cpu.built_with.is_empty() {
        parts.push(CpuFeature::join_labels(&cpu.built_with));
    }
    let missing = cpu.missing_on_cpu();
    if !missing.is_empty() {
        parts.push(format!(
            "not on this CPU: {}",
            CpuFeature::join_labels(&missing)
        ));
    }
    let unused = cpu.unused_by_build();
    if !unused.is_empty() {
        parts.push(format!("unused: {}", CpuFeature::join_labels(&unused)));
    }
    (!parts.is_empty()).then(|| parts.join(" · "))
}

fn memory_detail(device: &DeviceInfo) -> Option<String> {
    let total = device.memory_total.filter(|total| *total > 0)?;
    Some(format!("{} memory", ByteSize(total as u64)))
}

fn available_rows(report: &BackendReport, cpu: &CpuReport) -> Vec<SystemRow> {
    let backend = report.backend.label();
    if report.devices.is_empty() {
        return vec![SystemRow {
            backend,
            name: "The system's default device".to_owned(),
            detail: (report.backend == AccelBackend::Cpu)
                .then(|| cpu_detail(cpu))
                .flatten(),
            badge: Badge::Available,
        }];
    }
    report
        .devices
        .iter()
        .map(|device| SystemRow {
            backend,
            name: device_name(device),
            detail: if report.backend == AccelBackend::Cpu {
                cpu_detail(cpu)
            } else {
                memory_detail(device)
            },
            badge: Badge::Available,
        })
        .collect()
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

fn unavailable_row(backend: AccelBackend, why: &BackendUnavailable) -> SystemRow {
    let (name, detail, badge) = match why {
        BackendUnavailable::NotBuilt => {
            ("Not part of this build".to_owned(), None, Badge::NotBuilt)
        }
        BackendUnavailable::NoDevice => (
            "No usable device found".to_owned(),
            Some("Purr supports it, but no device or driver responded.".to_owned()),
            Badge::BuiltUnavailable,
        ),
        BackendUnavailable::RuntimeLibraryMissing { library, searched } => (
            format!("Missing {library}"),
            Some(missing_library_detail(searched)),
            Badge::BuiltUnavailable,
        ),
        BackendUnavailable::NotBuiltOrNoDevice => (
            "Not built in, or no device found".to_owned(),
            None,
            Badge::Unavailable,
        ),
    };
    SystemRow {
        backend: backend.label(),
        name,
        detail,
        badge,
    }
}

/// The panel's rows: every backend in priority order, a row per device for the ones that run.
#[must_use]
pub fn rows(report: &SystemReport) -> Vec<SystemRow> {
    report
        .backends
        .iter()
        .flat_map(|entry| match &entry.status {
            BackendStatus::Available => available_rows(entry, &report.cpu),
            BackendStatus::Unavailable(why) => vec![unavailable_row(entry.backend, why)],
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
                "Purr cannot transcribe here. Use a current Chrome, Edge or another browser with WebGPU enabled."
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

/// What to say about the last transcription's backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LastRunNote {
    pub fell_back: bool,
    pub title: String,
    pub detail: String,
}

/// The note under the panel for the last transcription; `None` when none ran yet.
///
/// A backend that is routinely skipped, because the build lacks it or the machine has no device
/// for it, is mentioned only when the user asked for it by name (`requested`).
///
/// This is the same rule as the banner on a run, so the two never disagree.
#[must_use]
pub fn last_run_note(
    last: Option<&LastRun>,
    requested: Option<AccelBackend>,
) -> Option<LastRunNote> {
    let last = last?;
    let used = last.device.as_ref().map_or_else(
        || last.backend.label().to_owned(),
        |device| format!("{} on {device}", last.backend.label()),
    );
    let skipped: Vec<_> = last
        .passed_over
        .iter()
        .filter(|passed| !routinely_skipped(&passed.reason) || requested == Some(passed.backend))
        .map(|passed| {
            format!(
                "{} was passed over: {}",
                passed.backend.label(),
                why_passed_over(&passed.reason)
            )
        })
        .collect();
    Some(LastRunNote {
        fell_back: !skipped.is_empty(),
        title: format!("The last transcription ran on {used}"),
        detail: if skipped.is_empty() {
            String::new()
        } else {
            format!("{}.", skipped.join("; "))
        },
    })
}

/// The entries of the backend menu, marking the ones this machine cannot run.
#[must_use]
pub fn backend_options(report: Option<&SystemReport>) -> Vec<(AccelSelection, String)> {
    AccelSelection::ALL
        .into_iter()
        .map(|selection| {
            let unavailable = selection
                .backend()
                .zip(report)
                .is_some_and(|(backend, report)| {
                    report.backends.iter().any(|entry| {
                        entry.backend == backend && entry.status != BackendStatus::Available
                    })
                });
            let label = if unavailable {
                format!("{} (unavailable)", selection.label())
            } else {
                selection.label().to_owned()
            };
            (selection, label)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::accel::{CpuFeature, PassedOver, PassedReason};
    use purr_common::platform::DeviceType;

    fn device(
        name: &str,
        description: &str,
        kind: DeviceType,
        memory: Option<usize>,
    ) -> DeviceInfo {
        DeviceInfo {
            id: 0,
            name: name.to_owned(),
            description: Some(description.to_owned()),
            device_type: kind,
            memory_free: None,
            memory_total: memory,
            capabilities: None,
        }
    }

    fn backend(
        backend: AccelBackend,
        status: BackendStatus,
        devices: Vec<DeviceInfo>,
    ) -> BackendReport {
        BackendReport {
            backend,
            status,
            devices,
        }
    }

    fn report() -> SystemReport {
        SystemReport {
            backends: vec![
                backend(
                    AccelBackend::Cuda,
                    BackendStatus::Unavailable(BackendUnavailable::RuntimeLibraryMissing {
                        library: "cudart64_13.dll".to_owned(),
                        searched: vec!["/opt/purr/runtime".to_owned(), "/opt/purr".to_owned()],
                    }),
                    vec![],
                ),
                backend(
                    AccelBackend::Metal,
                    BackendStatus::Unavailable(BackendUnavailable::NotBuilt),
                    vec![],
                ),
                backend(
                    AccelBackend::Sycl,
                    BackendStatus::Unavailable(BackendUnavailable::NotBuiltOrNoDevice),
                    vec![],
                ),
                backend(
                    AccelBackend::Vulkan,
                    BackendStatus::Available,
                    vec![
                        device(
                            "Vulkan0",
                            "NVIDIA GeForce RTX 4070",
                            DeviceType::Gpu,
                            Some(12 << 30),
                        ),
                        device("Vulkan1", "Intel UHD Graphics 770", DeviceType::Gpu, None),
                    ],
                ),
                backend(
                    AccelBackend::Hip,
                    BackendStatus::Unavailable(BackendUnavailable::NoDevice),
                    vec![],
                ),
                backend(
                    AccelBackend::Cpu,
                    BackendStatus::Available,
                    vec![device(
                        "CPU",
                        "Some Processor 9000",
                        DeviceType::Cpu,
                        Some(0),
                    )],
                ),
            ],
            cpu: CpuReport {
                name: None,
                logical_threads: Some(24),
                built_with: vec![CpuFeature::Avx2, CpuFeature::Fma],
                detected: Some(vec![CpuFeature::Avx2, CpuFeature::Fma]),
            },
            last_run: None,
        }
    }

    fn cpu(built_with: &[CpuFeature], detected: Option<&[CpuFeature]>) -> CpuReport {
        CpuReport {
            name: None,
            logical_threads: Some(8),
            built_with: built_with.to_vec(),
            detected: detected.map(<[CpuFeature]>::to_vec),
        }
    }

    #[test]
    fn the_cpu_detail_lists_what_the_build_uses() {
        let both = [CpuFeature::Avx2, CpuFeature::Fma];
        assert_eq!(
            cpu_detail(&cpu(&both, Some(&both))),
            Some("8 threads · AVX2, FMA".to_owned())
        );
    }

    #[test]
    fn the_cpu_detail_flags_a_mismatch_between_the_build_and_the_processor() {
        assert_eq!(
            cpu_detail(&cpu(
                &[CpuFeature::Avx2, CpuFeature::Avx512F],
                Some(&[CpuFeature::Avx2, CpuFeature::Bmi2])
            )),
            Some("8 threads · AVX2, AVX512F · not on this CPU: AVX512F · unused: BMI2".to_owned())
        );
    }

    #[test]
    fn the_cpu_detail_makes_no_claim_where_the_processor_cannot_be_probed() {
        assert_eq!(
            cpu_detail(&cpu(&[CpuFeature::Avx2], None)),
            Some("8 threads · AVX2".to_owned())
        );
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
    fn every_backend_gets_a_row_with_its_status_and_devices_get_their_own() {
        assert_eq!(
            rows(&report()),
            [
                row(
                    "CUDA",
                    "Missing cudart64_13.dll",
                    Some(
                        "Purr supports it, but the library it needs is not installed. Looked in /opt/purr/runtime, /opt/purr, then the system's library folders."
                    ),
                    Badge::BuiltUnavailable
                ),
                row("Metal", "Not part of this build", None, Badge::NotBuilt),
                row(
                    "SYCL",
                    "Not built in, or no device found",
                    None,
                    Badge::Unavailable
                ),
                row(
                    "Vulkan",
                    "NVIDIA GeForce RTX 4070",
                    Some("12.0 GiB memory"),
                    Badge::Available
                ),
                row("Vulkan", "Intel UHD Graphics 770", None, Badge::Available),
                row(
                    "ROCm",
                    "No usable device found",
                    Some("Purr supports it, but no device or driver responded."),
                    Badge::BuiltUnavailable
                ),
                row(
                    "CPU",
                    "Some Processor 9000",
                    Some("24 threads · AVX2, FMA"),
                    Badge::Available
                ),
            ]
        );
    }

    #[test]
    fn an_available_backend_without_a_listed_device_still_gets_a_row() {
        let report = SystemReport {
            backends: vec![backend(
                AccelBackend::CoreMl,
                BackendStatus::Available,
                vec![],
            )],
            cpu: CpuReport {
                name: None,
                logical_threads: None,
                built_with: vec![],
                detected: None,
            },
            last_run: None,
        };
        assert_eq!(
            rows(&report),
            [row(
                "CoreML",
                "The system's default device",
                None,
                Badge::Available
            )]
        );
    }

    #[test]
    fn the_menu_marks_the_backends_the_machine_cannot_run() {
        let options = backend_options(Some(&report()));
        let marked: Vec<_> = options
            .iter()
            .filter(|(_, label)| label.ends_with("(unavailable)"))
            .map(|(selection, _)| *selection)
            .collect();
        assert_eq!(
            marked,
            [
                AccelSelection::Cuda,
                AccelSelection::Hip,
                AccelSelection::Metal,
                AccelSelection::Sycl
            ]
        );
        assert_eq!(
            options.first().map(|(_, label)| label.as_str()),
            Some("Automatic")
        );
    }

    #[test]
    fn the_menu_marks_nothing_before_the_probe_finished() {
        let options = backend_options(None);
        assert_eq!(options.len(), AccelSelection::ALL.len());
        assert!(
            options
                .iter()
                .all(|(selection, label)| label == selection.label())
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

    #[test]
    fn no_note_before_a_transcription_ran() {
        assert_eq!(last_run_note(None, None), None);
    }

    #[test]
    fn a_clean_run_names_the_backend_and_device() {
        let note = last_run_note(
            Some(&LastRun {
                backend: AccelBackend::Vulkan,
                device: Some("NVIDIA GeForce RTX 4070".to_owned()),
                passed_over: vec![],
            }),
            None,
        );
        assert_eq!(
            note,
            Some(LastRunNote {
                fell_back: false,
                title: "The last transcription ran on Vulkan on NVIDIA GeForce RTX 4070".to_owned(),
                detail: String::new(),
            })
        );
    }

    #[test]
    fn a_run_that_passed_over_a_backend_says_why() {
        let note = last_run_note(
            Some(&LastRun {
                backend: AccelBackend::Cpu,
                device: None,
                passed_over: vec![
                    PassedOver {
                        backend: AccelBackend::Cuda,
                        reason: PassedReason::Unavailable(
                            BackendUnavailable::RuntimeLibraryMissing {
                                library: "cudart64_13.dll".to_owned(),
                                searched: Vec::new(),
                            },
                        ),
                    },
                    PassedOver {
                        backend: AccelBackend::Metal,
                        reason: PassedReason::Unavailable(BackendUnavailable::NotBuilt),
                    },
                    PassedOver {
                        backend: AccelBackend::Vulkan,
                        reason: PassedReason::Failed("device 0 failed to initialise".to_owned()),
                    },
                ],
            }),
            None,
        );
        assert_eq!(
            note,
            Some(LastRunNote {
                fell_back: true,
                title: "The last transcription ran on CPU".to_owned(),
                detail: "CUDA was passed over: cudart64_13.dll is missing; Vulkan was passed over: device 0 failed to initialise.".to_owned(),
            })
        );
    }

    #[test]
    fn backends_that_are_not_part_of_the_build_are_not_news() {
        let last = LastRun {
            backend: AccelBackend::Cpu,
            device: None,
            passed_over: vec![PassedOver {
                backend: AccelBackend::Metal,
                reason: PassedReason::Unavailable(BackendUnavailable::NotBuilt),
            }],
        };
        assert_eq!(
            last_run_note(Some(&last), None).map(|note| note.fell_back),
            Some(false)
        );
        assert_eq!(
            last_run_note(Some(&last), Some(AccelBackend::Vulkan)).map(|note| note.fell_back),
            Some(false)
        );
    }

    #[test]
    fn a_backend_the_user_asked_for_is_reported_even_when_it_is_not_in_the_build() {
        let last = LastRun {
            backend: AccelBackend::Cpu,
            device: None,
            passed_over: vec![
                PassedOver {
                    backend: AccelBackend::Cuda,
                    reason: PassedReason::Unavailable(BackendUnavailable::NotBuilt),
                },
                PassedOver {
                    backend: AccelBackend::Vulkan,
                    reason: PassedReason::Unavailable(BackendUnavailable::NotBuilt),
                },
            ],
        };
        assert_eq!(
            last_run_note(Some(&last), Some(AccelBackend::Vulkan)),
            Some(LastRunNote {
                fell_back: true,
                title: "The last transcription ran on CPU".to_owned(),
                detail: "Vulkan was passed over: it is not part of this build.".to_owned(),
            })
        );
    }

    #[test]
    fn a_backend_without_a_device_is_not_news_unless_the_user_asked_for_it() {
        let last = LastRun {
            backend: AccelBackend::Cpu,
            device: None,
            passed_over: vec![PassedOver {
                backend: AccelBackend::Cuda,
                reason: PassedReason::Unavailable(BackendUnavailable::NoDevice),
            }],
        };
        assert_eq!(
            last_run_note(Some(&last), None).map(|note| note.fell_back),
            Some(false)
        );
        assert_eq!(
            last_run_note(Some(&last), Some(AccelBackend::Cuda)).map(|note| note.detail),
            Some("CUDA was passed over: no usable device responded.".to_owned())
        );
    }
}
