//! Picks the execution provider for a model from the settings, and moves a run to the CPU when
//! WebGPU cannot start. Nothing is measured: the GPU is trusted until the worker says it failed.

use crate::protocol::{FailureKind, Provider};
use purr_common::accel::AccelPolicy;
use purr_common::browser_engine::{BrowserEngine, CpuReason};
use purr_common::web_model::registry::WebModelSpec;

/// Where a model runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice {
    pub engine: BrowserEngine,
    cpu_can_run: bool,
}

/// What a worker failure means for a run on a [`Choice`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuFailure {
    /// The run goes on, on the CPU.
    MoveToCpu(Choice),
    /// WebGPU failed and the model cannot run on the CPU.
    CannotRun(CpuReason),
    /// The failure has nothing to do with WebGPU, or the run was not on it.
    Unrelated,
}

impl Choice {
    #[must_use]
    pub const fn provider(&self) -> Provider {
        match self.engine {
            BrowserEngine::WebGpu => Provider::WebGpu,
            BrowserEngine::Cpu(_) => Provider::Cpu,
        }
    }

    #[must_use]
    pub fn after_failure(&self, kind: FailureKind) -> GpuFailure {
        match (self.engine, gpu_failure_reason(kind)) {
            (BrowserEngine::WebGpu, Some(reason)) if self.cpu_can_run => {
                GpuFailure::MoveToCpu(Self {
                    engine: BrowserEngine::Cpu(reason),
                    cpu_can_run: true,
                })
            }
            (BrowserEngine::WebGpu, Some(reason)) => GpuFailure::CannotRun(reason),
            (BrowserEngine::WebGpu, None) | (BrowserEngine::Cpu(_), _) => GpuFailure::Unrelated,
        }
    }
}

/// The CPU reason a worker failure gives, for failures that come from WebGPU.
#[must_use]
pub const fn gpu_failure_reason(kind: FailureKind) -> Option<CpuReason> {
    match kind {
        FailureKind::WebGpuUnavailable => Some(CpuReason::NoWebGpu),
        FailureKind::ShaderF16Unavailable => Some(CpuReason::NoShaderF16),
        FailureKind::WebGpuFailed => Some(CpuReason::WebGpuFailed),
        FailureKind::ModelLoad
        | FailureKind::Inference
        | FailureKind::Protocol
        | FailureKind::WorkerCrashed
        | FailureKind::MalformedEvent
        | FailureKind::WorkerRestarted => None,
    }
}

/// Where `spec` runs under `policy`, or why it cannot: it only runs on WebGPU and the settings
/// ask for the CPU.
pub fn choose(spec: &WebModelSpec, policy: AccelPolicy) -> Result<Choice, CpuReason> {
    let cpu_can_run = spec.runs_on_cpu();
    match BrowserEngine::requested(policy) {
        BrowserEngine::WebGpu => Ok(Choice {
            engine: BrowserEngine::WebGpu,
            cpu_can_run,
        }),
        BrowserEngine::Cpu(reason) if cpu_can_run => Ok(Choice {
            engine: BrowserEngine::Cpu(reason),
            cpu_can_run,
        }),
        BrowserEngine::Cpu(reason) => Err(reason),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::web_model::registry::WebModelRegistry;

    const CPU_RUNNABLE: &str = "base-q4";
    const GPU_ONLY: &str = "large-v3-turbo";

    fn spec(id: &str) -> Result<WebModelSpec, Box<dyn std::error::Error>> {
        Ok(WebModelRegistry::embedded()?.get(id)?.clone())
    }

    fn chosen(id: &str, policy: AccelPolicy) -> Result<Choice, Box<dyn std::error::Error>> {
        choose(&spec(id)?, policy).map_err(|reason| format!("{reason:?}").into())
    }

    #[test]
    fn the_default_policy_runs_on_webgpu() -> Result<(), Box<dyn std::error::Error>> {
        for id in [CPU_RUNNABLE, GPU_ONLY] {
            assert_eq!(
                choose(&spec(id)?, AccelPolicy::Auto).map(|choice| choice.engine),
                Ok(BrowserEngine::WebGpu),
                "{id}"
            );
        }
        Ok(())
    }

    #[test]
    fn the_cpu_policy_runs_on_the_cpu_when_the_model_can() -> Result<(), Box<dyn std::error::Error>>
    {
        assert_eq!(
            choose(&spec(CPU_RUNNABLE)?, AccelPolicy::CPU_ONLY).map(|choice| choice.engine),
            Ok(BrowserEngine::Cpu(CpuReason::Chosen))
        );
        assert_eq!(
            choose(&spec(GPU_ONLY)?, AccelPolicy::CPU_ONLY).map(|choice| choice.engine),
            Err(CpuReason::Chosen)
        );
        Ok(())
    }

    #[test]
    fn the_provider_follows_the_engine() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            chosen(CPU_RUNNABLE, AccelPolicy::Auto)?.provider(),
            Provider::WebGpu
        );
        assert_eq!(
            chosen(CPU_RUNNABLE, AccelPolicy::CPU_ONLY)?.provider(),
            Provider::Cpu
        );
        Ok(())
    }

    #[test]
    fn a_webgpu_failure_moves_a_cpu_capable_model_to_the_cpu()
    -> Result<(), Box<dyn std::error::Error>> {
        let choice = chosen(CPU_RUNNABLE, AccelPolicy::Auto)?;
        for (kind, reason) in [
            (FailureKind::WebGpuUnavailable, CpuReason::NoWebGpu),
            (FailureKind::ShaderF16Unavailable, CpuReason::NoShaderF16),
            (FailureKind::WebGpuFailed, CpuReason::WebGpuFailed),
        ] {
            assert_eq!(
                choice.after_failure(kind),
                GpuFailure::MoveToCpu(Choice {
                    engine: BrowserEngine::Cpu(reason),
                    cpu_can_run: true,
                }),
                "{kind:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn a_webgpu_failure_refuses_a_model_the_cpu_cannot_run()
    -> Result<(), Box<dyn std::error::Error>> {
        let choice = chosen(GPU_ONLY, AccelPolicy::Auto)?;
        assert_eq!(
            choice.after_failure(FailureKind::WebGpuUnavailable),
            GpuFailure::CannotRun(CpuReason::NoWebGpu)
        );
        Ok(())
    }

    #[test]
    fn other_failures_and_runs_already_on_the_cpu_stay_as_they_are()
    -> Result<(), Box<dyn std::error::Error>> {
        let gpu = chosen(CPU_RUNNABLE, AccelPolicy::Auto)?;
        let cpu = chosen(CPU_RUNNABLE, AccelPolicy::CPU_ONLY)?;
        assert_eq!(
            gpu.after_failure(FailureKind::Inference),
            GpuFailure::Unrelated
        );
        assert_eq!(
            gpu.after_failure(FailureKind::ModelLoad),
            GpuFailure::Unrelated
        );
        assert_eq!(
            cpu.after_failure(FailureKind::WebGpuFailed),
            GpuFailure::Unrelated
        );
        Ok(())
    }
}
