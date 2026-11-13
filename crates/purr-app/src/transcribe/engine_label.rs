//! The engine as every screen words it: the backend and the hardware it runs on.

use super::engine::RunMemory;
use crate::hooks::{use_models::GpuSupport, use_system_probe::SystemProbe};
use purr_common::{
    accel::{
        AccelBackend, AccelPolicy, BackendReport, BackendStatus, EngineUsed, LastRun, SystemReport,
    },
    browser_engine::BrowserEngine,
    platform::{DeviceInfo, DeviceType},
};
use std::fmt;

/// What runs the transcription, as far as it is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendLabel {
    Accel(AccelBackend),
    WebGpu,
    /// The settings leave the choice to Purr and nothing says what it will pick.
    Automatic,
    /// The machine is still being probed.
    Checking,
}

impl BackendLabel {
    const fn text(self) -> &'static str {
        match self {
            Self::Accel(backend) => backend.label(),
            Self::WebGpu => "WebGPU",
            Self::Automatic => "Automatic",
            Self::Checking => "Checking…",
        }
    }
}

impl fmt::Display for BackendLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.text())
    }
}

/// The backend and the hardware name that go with it, when one is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineLabel {
    pub backend: BackendLabel,
    pub hardware: Option<String>,
}

impl fmt::Display for EngineLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.hardware {
            Some(hardware) => write!(f, "{} · {hardware}", self.backend),
            None => write!(f, "{}", self.backend),
        }
    }
}

fn report(probe: &SystemProbe) -> Option<&SystemReport> {
    match probe {
        SystemProbe::Ready(report) => Some(report),
        SystemProbe::Checking | SystemProbe::Unavailable => None,
    }
}

fn entry(report: &SystemReport, backend: AccelBackend) -> Option<&BackendReport> {
    report
        .backends
        .iter()
        .find(|entry| entry.backend == backend)
}

fn is_available(report: &SystemReport, backend: AccelBackend) -> bool {
    entry(report, backend).is_some_and(|entry| entry.status == BackendStatus::Available)
}

fn best_available(report: &SystemReport) -> AccelBackend {
    AccelBackend::PRIORITY
        .into_iter()
        .find(|backend| is_available(report, *backend))
        .unwrap_or(AccelBackend::Cpu)
}

fn device_name(device: &DeviceInfo) -> Option<String> {
    [device.description.as_deref(), Some(device.name.as_str())]
        .into_iter()
        .flatten()
        .map(str::trim)
        .find(|name| !name.is_empty())
        .map(str::to_owned)
}

fn hardware(
    backend: AccelBackend,
    report: &SystemReport,
    ran_device: Option<&str>,
) -> Option<String> {
    match backend {
        AccelBackend::Cpu => report.cpu.name.clone(),
        AccelBackend::CoreMl => None,
        AccelBackend::Cuda
        | AccelBackend::Hip
        | AccelBackend::Metal
        | AccelBackend::Sycl
        | AccelBackend::Vulkan => {
            let devices = &entry(report, backend)?.devices;
            ran_device
                .and_then(|name| devices.iter().find(|device| device.name == name))
                .or_else(|| {
                    devices
                        .iter()
                        .find(|device| device.device_type == DeviceType::Gpu)
                })
                .and_then(device_name)
        }
    }
}

fn expected_backend(policy: AccelPolicy, report: Option<&SystemReport>) -> Option<AccelBackend> {
    match (policy, report) {
        (AccelPolicy::CPU_ONLY, _) => Some(AccelBackend::Cpu),
        (
            AccelPolicy::Forced {
                backend,
                allow_fallback: false,
            },
            _,
        )
        | (
            AccelPolicy::Forced {
                backend,
                allow_fallback: true,
            },
            None,
        ) => Some(backend),
        (
            AccelPolicy::Forced {
                backend,
                allow_fallback: true,
            },
            Some(report),
        ) if is_available(report, backend) => Some(backend),
        (AccelPolicy::Forced { .. } | AccelPolicy::Auto, Some(report)) => {
            Some(best_available(report))
        }
        (AccelPolicy::Auto, None) => None,
    }
}

impl EngineLabel {
    /// What a saved run ran on.
    #[must_use]
    pub fn saved(engine: &EngineUsed) -> Self {
        Self {
            backend: BackendLabel::Accel(engine.backend),
            hardware: engine.hardware.clone(),
        }
    }

    /// What a run ended up on, as stored with its history entry.
    #[must_use]
    pub fn used(last: &LastRun, probe: &SystemProbe) -> EngineUsed {
        EngineUsed {
            backend: last.backend,
            hardware: Self::ran(last, probe).hardware,
        }
    }

    /// What a run ended up on.
    #[must_use]
    pub fn ran(last: &LastRun, probe: &SystemProbe) -> Self {
        Self {
            backend: BackendLabel::Accel(last.backend),
            hardware: report(probe)
                .and_then(|report| hardware(last.backend, report, last.device.as_deref())),
        }
    }

    /// What the next run is expected to use: the run made under the same policy, else what the
    /// runtime's priority order picks on this machine.
    #[must_use]
    pub fn planned(policy: AccelPolicy, memory: Option<&RunMemory>, probe: &SystemProbe) -> Self {
        if let Some(memory) = memory.filter(|memory| memory.policy == policy) {
            return Self::ran(&memory.ran, probe);
        }
        let report = report(probe);
        expected_backend(policy, report).map_or_else(
            || Self {
                backend: match probe {
                    SystemProbe::Checking => BackendLabel::Checking,
                    SystemProbe::Unavailable | SystemProbe::Ready(_) => BackendLabel::Automatic,
                },
                hardware: None,
            },
            |backend| Self {
                backend: BackendLabel::Accel(backend),
                hardware: report.and_then(|report| hardware(backend, report, None)),
            },
        )
    }

    /// What a browser run uses: the WebGPU adapter, or the CPU.
    #[must_use]
    pub fn browser(engine: BrowserEngine, gpu: &GpuSupport) -> Self {
        match engine {
            BrowserEngine::WebGpu => Self {
                backend: BackendLabel::WebGpu,
                hardware: match gpu {
                    GpuSupport::Available { name, .. } => name.clone(),
                    GpuSupport::Checking | GpuSupport::Unavailable => None,
                },
            },
            BrowserEngine::Cpu(_) => Self {
                backend: BackendLabel::Accel(AccelBackend::Cpu),
                hardware: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::accel::{CpuReport, PassedOver, PassedReason};
    use purr_common::browser_engine::CpuReason;

    const CPU_NAME: &str = "AMD Ryzen 9 7950X 16-Core Processor";
    const GPU_NAME: &str = "NVIDIA GeForce RTX 4090";

    fn forced(backend: AccelBackend) -> AccelPolicy {
        AccelPolicy::Forced {
            backend,
            allow_fallback: true,
        }
    }

    fn device(name: &str, description: &str, device_type: DeviceType) -> DeviceInfo {
        DeviceInfo {
            id: 0,
            name: name.to_owned(),
            description: Some(description.to_owned()),
            device_type,
            memory_free: None,
            memory_total: None,
            capabilities: None,
        }
    }

    fn available(backend: AccelBackend, devices: Vec<DeviceInfo>) -> BackendReport {
        BackendReport {
            backend,
            status: BackendStatus::Available,
            devices,
        }
    }

    fn missing(backend: AccelBackend) -> BackendReport {
        BackendReport {
            backend,
            status: BackendStatus::Unavailable(purr_common::accel::BackendUnavailable::NoDevice),
            devices: vec![],
        }
    }

    fn machine(cuda: bool) -> SystemProbe {
        SystemProbe::Ready(SystemReport {
            backends: vec![
                if cuda {
                    available(
                        AccelBackend::Cuda,
                        vec![device("CUDA0", GPU_NAME, DeviceType::Gpu)],
                    )
                } else {
                    missing(AccelBackend::Cuda)
                },
                missing(AccelBackend::Metal),
                available(
                    AccelBackend::Vulkan,
                    vec![
                        device("Vulkan0", "Intel UHD Graphics 770", DeviceType::Gpu),
                        device("Vulkan1", GPU_NAME, DeviceType::Gpu),
                    ],
                ),
                available(
                    AccelBackend::Cpu,
                    vec![device("CPU", CPU_NAME, DeviceType::Cpu)],
                ),
            ],
            cpu: CpuReport {
                name: Some(CPU_NAME.to_owned()),
                logical_threads: Some(32),
                built_with: vec![],
                detected: None,
            },
            last_run: None,
        })
    }

    fn label(backend: AccelBackend, hardware: Option<&str>) -> EngineLabel {
        EngineLabel {
            backend: BackendLabel::Accel(backend),
            hardware: hardware.map(str::to_owned),
        }
    }

    fn ran(backend: AccelBackend, device: Option<&str>) -> LastRun {
        LastRun {
            backend,
            device: device.map(str::to_owned),
            passed_over: vec![],
        }
    }

    #[test]
    fn automatic_predicts_the_best_available_backend_and_its_gpu() {
        assert_eq!(
            EngineLabel::planned(AccelPolicy::Auto, None, &machine(true)),
            label(AccelBackend::Cuda, Some(GPU_NAME))
        );
        assert_eq!(
            EngineLabel::planned(AccelPolicy::Auto, None, &machine(false)),
            label(AccelBackend::Vulkan, Some("Intel UHD Graphics 770"))
        );
    }

    #[test]
    fn the_processor_only_names_the_processor() {
        assert_eq!(
            EngineLabel::planned(AccelPolicy::CPU_ONLY, None, &machine(true)),
            label(AccelBackend::Cpu, Some(CPU_NAME))
        );
    }

    #[test]
    fn a_named_backend_is_what_the_settings_say() {
        assert_eq!(
            EngineLabel::planned(forced(AccelBackend::Vulkan), None, &machine(true)),
            label(AccelBackend::Vulkan, Some("Intel UHD Graphics 770"))
        );
    }

    #[test]
    fn a_named_backend_that_cannot_run_falls_through_when_fallback_is_allowed() {
        assert_eq!(
            EngineLabel::planned(forced(AccelBackend::Cuda), None, &machine(false)),
            label(AccelBackend::Vulkan, Some("Intel UHD Graphics 770"))
        );
    }

    #[test]
    fn a_named_backend_that_cannot_run_stays_named_without_fallback() {
        let policy = AccelPolicy::Forced {
            backend: AccelBackend::Cuda,
            allow_fallback: false,
        };
        assert_eq!(
            EngineLabel::planned(policy, None, &machine(false)),
            label(AccelBackend::Cuda, None)
        );
    }

    #[test]
    fn the_last_run_under_the_same_policy_says_what_will_run() {
        let policy = forced(AccelBackend::Cuda);
        let memory = RunMemory {
            policy,
            ran: LastRun {
                backend: AccelBackend::Vulkan,
                device: Some("Vulkan1".to_owned()),
                passed_over: vec![PassedOver {
                    backend: AccelBackend::Cuda,
                    reason: PassedReason::Failed("the driver is too old".to_owned()),
                }],
            },
        };
        assert_eq!(
            EngineLabel::planned(policy, Some(&memory), &machine(true)),
            label(AccelBackend::Vulkan, Some(GPU_NAME))
        );
    }

    #[test]
    fn a_run_made_under_another_policy_says_nothing() {
        let memory = RunMemory {
            policy: AccelPolicy::CPU_ONLY,
            ran: ran(AccelBackend::Cpu, None),
        };
        assert_eq!(
            EngineLabel::planned(AccelPolicy::Auto, Some(&memory), &machine(true)),
            label(AccelBackend::Cuda, Some(GPU_NAME))
        );
    }

    #[test]
    fn a_run_is_named_after_the_device_it_used() {
        assert_eq!(
            EngineLabel::ran(&ran(AccelBackend::Vulkan, Some("Vulkan1")), &machine(false)),
            label(AccelBackend::Vulkan, Some(GPU_NAME))
        );
    }

    #[test]
    fn a_run_on_the_processor_is_named_after_it() {
        assert_eq!(
            EngineLabel::ran(&ran(AccelBackend::Cpu, None), &machine(true)),
            label(AccelBackend::Cpu, Some(CPU_NAME))
        );
    }

    #[test]
    fn a_device_the_report_lacks_leaves_the_backend_alone_or_the_first_gpu() {
        assert_eq!(
            EngineLabel::ran(&ran(AccelBackend::Vulkan, Some("Vulkan9")), &machine(true)),
            label(AccelBackend::Vulkan, Some("Intel UHD Graphics 770"))
        );
        assert_eq!(
            EngineLabel::ran(&ran(AccelBackend::Metal, None), &machine(true)),
            label(AccelBackend::Metal, None)
        );
    }

    #[test]
    fn without_a_report_the_backend_stands_alone() {
        assert_eq!(
            EngineLabel::ran(
                &ran(AccelBackend::Vulkan, Some("Vulkan0")),
                &SystemProbe::Unavailable
            ),
            label(AccelBackend::Vulkan, None)
        );
        assert_eq!(
            EngineLabel::planned(forced(AccelBackend::Cuda), None, &SystemProbe::Checking),
            label(AccelBackend::Cuda, None)
        );
    }

    #[test]
    fn automatic_waits_for_the_probe() {
        assert_eq!(
            EngineLabel::planned(AccelPolicy::Auto, None, &SystemProbe::Checking),
            EngineLabel {
                backend: BackendLabel::Checking,
                hardware: None
            }
        );
        assert_eq!(
            EngineLabel::planned(AccelPolicy::Auto, None, &SystemProbe::Unavailable),
            EngineLabel {
                backend: BackendLabel::Automatic,
                hardware: None
            }
        );
    }

    fn adapter(name: Option<&str>) -> GpuSupport {
        GpuSupport::Available {
            name: name.map(str::to_owned),
            shader_f16: true,
        }
    }

    #[test]
    fn the_browser_shows_webgpu_and_whatever_the_adapter_reports() {
        assert_eq!(
            EngineLabel::browser(BrowserEngine::WebGpu, &adapter(Some("NVIDIA (Ampere)"))),
            EngineLabel {
                backend: BackendLabel::WebGpu,
                hardware: Some("NVIDIA (Ampere)".to_owned())
            }
        );
        for gpu in [adapter(None), GpuSupport::Checking, GpuSupport::Unavailable] {
            assert_eq!(
                EngineLabel::browser(BrowserEngine::WebGpu, &gpu),
                EngineLabel {
                    backend: BackendLabel::WebGpu,
                    hardware: None
                }
            );
        }
    }

    #[test]
    fn a_browser_run_on_the_cpu_says_cpu_and_nothing_else() {
        assert_eq!(
            EngineLabel::browser(
                BrowserEngine::Cpu(CpuReason::WebGpuFailed),
                &adapter(Some("NVIDIA (Ampere)"))
            ),
            EngineLabel {
                backend: BackendLabel::Accel(AccelBackend::Cpu),
                hardware: None
            }
        );
    }

    #[test]
    fn the_text_is_the_backend_then_the_hardware() {
        assert_eq!(
            label(AccelBackend::Vulkan, Some(GPU_NAME)).to_string(),
            "Vulkan · NVIDIA GeForce RTX 4090"
        );
        assert_eq!(label(AccelBackend::Cpu, None).to_string(), "CPU");
    }
}
