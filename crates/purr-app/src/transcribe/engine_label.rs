//! A finished run's engine as the screens word it: the backend and the hardware it ran on.

use crate::hooks::use_system_probe::SystemProbe;
use purr_common::accel::{AccelBackend, EngineUsed, LastRun};
use std::fmt;

/// The backend and the hardware name that go with it, when one is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineLabel {
    pub backend: AccelBackend,
    pub hardware: Option<String>,
}

impl fmt::Display for EngineLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.hardware {
            Some(hardware) => write!(f, "{} · {hardware}", self.backend.label()),
            None => f.write_str(self.backend.label()),
        }
    }
}

impl EngineLabel {
    /// What a saved run ran on.
    #[must_use]
    pub fn saved(engine: &EngineUsed) -> Self {
        Self {
            backend: engine.backend,
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

    /// What a run ended up on: a GPU by the name the run reported, the processor by the name
    /// the system probe gives.
    #[must_use]
    pub fn ran(last: &LastRun, probe: &SystemProbe) -> Self {
        let hardware = match last.backend {
            AccelBackend::Cpu => match probe {
                SystemProbe::Ready(report) => report.cpu.name.clone(),
                SystemProbe::Checking | SystemProbe::Unavailable => None,
            },
            AccelBackend::Cuda
            | AccelBackend::Hip
            | AccelBackend::CoreMl
            | AccelBackend::Metal
            | AccelBackend::Sycl
            | AccelBackend::Vulkan => last.device.clone(),
        };
        Self {
            backend: last.backend,
            hardware,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::accel::{CpuReport, SystemReport};

    const CPU_NAME: &str = "AMD Ryzen 9 7950X 16-Core Processor";
    const GPU_NAME: &str = "NVIDIA GeForce RTX 4090";

    fn machine() -> SystemProbe {
        SystemProbe::Ready(SystemReport {
            backends: vec![],
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
            backend,
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
    fn a_gpu_run_is_named_after_the_device_it_reported() {
        assert_eq!(
            EngineLabel::ran(&ran(AccelBackend::Vulkan, Some(GPU_NAME)), &machine()),
            label(AccelBackend::Vulkan, Some(GPU_NAME))
        );
        assert_eq!(
            EngineLabel::ran(
                &ran(AccelBackend::Cuda, Some(GPU_NAME)),
                &SystemProbe::Unavailable
            ),
            label(AccelBackend::Cuda, Some(GPU_NAME))
        );
    }

    #[test]
    fn a_run_on_the_processor_is_named_after_it() {
        assert_eq!(
            EngineLabel::ran(&ran(AccelBackend::Cpu, None), &machine()),
            label(AccelBackend::Cpu, Some(CPU_NAME))
        );
        assert_eq!(
            EngineLabel::ran(&ran(AccelBackend::Cpu, None), &SystemProbe::Checking),
            label(AccelBackend::Cpu, None)
        );
    }

    #[test]
    fn a_saved_run_keeps_what_it_stored() {
        assert_eq!(
            EngineLabel::saved(&EngineUsed {
                backend: AccelBackend::Vulkan,
                hardware: Some(GPU_NAME.to_owned())
            }),
            label(AccelBackend::Vulkan, Some(GPU_NAME))
        );
    }

    #[test]
    fn the_stored_engine_follows_the_run() {
        assert_eq!(
            EngineLabel::used(&ran(AccelBackend::Cpu, None), &machine()),
            EngineUsed {
                backend: AccelBackend::Cpu,
                hardware: Some(CPU_NAME.to_owned())
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
