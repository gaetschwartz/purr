//! Picks the execution provider for a model from the settings, and moves a run to the CPU when
//! WebGPU cannot start. Nothing is measured: the GPU is trusted until the worker says it failed.

use crate::protocol::{FailureKind, Provider};
use purr_common::browser_compute::{BrowserCompute, CpuReason};
use purr_common::devices::DeviceOrder;
use purr_common::web_model::registry::{WebModel, WebModelSpec};
use purr_common::web_model::variant::{ShaderF16, VariantKind};

/// Where a model runs, and with which of its weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice {
    pub compute: BrowserCompute,
    pub variant: VariantKind,
    cpu_variant: Option<VariantKind>,
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
        match self.compute {
            BrowserCompute::WebGpu => Provider::WebGpu,
            BrowserCompute::Cpu(_) => Provider::Cpu,
        }
    }

    #[must_use]
    pub fn after_failure(&self, kind: FailureKind) -> GpuFailure {
        match (self.compute, gpu_failure_reason(kind), self.cpu_variant) {
            (BrowserCompute::WebGpu, Some(reason), Some(cpu_variant)) => {
                GpuFailure::MoveToCpu(Self {
                    compute: BrowserCompute::Cpu(reason),
                    variant: cpu_variant,
                    cpu_variant: Some(cpu_variant),
                })
            }
            (BrowserCompute::WebGpu, Some(reason), None) => GpuFailure::CannotRun(reason),
            (BrowserCompute::WebGpu, None, _) | (BrowserCompute::Cpu(_), _, _) => {
                GpuFailure::Unrelated
            }
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

/// Where `model` runs under `devices` and with which variant, or why it cannot: it only runs on
/// WebGPU and the settings ask for the CPU. WebGPU gets half precision where the adapter has
/// `shader-f16`; the CPU, and a run that falls back to it, always get full precision.
pub fn choose(
    model: &WebModel,
    devices: &DeviceOrder,
    shader_f16: ShaderF16,
) -> Result<Choice, CpuReason> {
    let cpu_variant = model.cpu_variant().map(|spec| spec.variant);
    match (BrowserCompute::requested(devices), cpu_variant) {
        (BrowserCompute::WebGpu, cpu_variant) => Ok(Choice {
            compute: BrowserCompute::WebGpu,
            variant: model.webgpu_variant(shader_f16).variant,
            cpu_variant,
        }),
        (BrowserCompute::Cpu(reason), Some(variant)) => Ok(Choice {
            compute: BrowserCompute::Cpu(reason),
            variant,
            cpu_variant,
        }),
        (BrowserCompute::Cpu(reason), None) => Err(reason),
    }
}

/// The variant whose installation and size `model` shows under `devices`: the one a run would use.
#[must_use]
pub fn shown_variant<'a>(
    model: &'a WebModel,
    devices: &DeviceOrder,
    shader_f16: ShaderF16,
) -> &'a WebModelSpec {
    choose(model, devices, shader_f16)
        .ok()
        .and_then(|choice| model.variant(choice.variant))
        .unwrap_or_else(|| model.baseline())
}

#[cfg(test)]
mod tests {
    use super::*;
    use purr_common::web_model::registry::WebModelRegistry;

    const CPU_RUNNABLE: &str = "base-q4";
    const GPU_ONLY: &str = "large-v3-turbo";
    const F16: ShaderF16 = ShaderF16::Supported;

    fn model(id: &str) -> Result<WebModel, Box<dyn std::error::Error>> {
        Ok(WebModelRegistry::embedded()?.get(id)?.clone())
    }

    fn chosen(
        id: &str,
        devices: &DeviceOrder,
        shader_f16: ShaderF16,
    ) -> Result<Choice, Box<dyn std::error::Error>> {
        choose(&model(id)?, devices, shader_f16).map_err(|reason| format!("{reason:?}").into())
    }

    #[test]
    fn the_default_policy_runs_on_webgpu() -> Result<(), Box<dyn std::error::Error>> {
        for id in [CPU_RUNNABLE, GPU_ONLY] {
            assert_eq!(
                choose(&model(id)?, &DeviceOrder::automatic(), F16).map(|choice| choice.compute),
                Ok(BrowserCompute::WebGpu),
                "{id}"
            );
        }
        Ok(())
    }

    #[test]
    fn the_cpu_policy_runs_on_the_cpu_when_the_model_can() -> Result<(), Box<dyn std::error::Error>>
    {
        assert_eq!(
            choose(&model(CPU_RUNNABLE)?, &DeviceOrder::cpu_only(), F16)
                .map(|choice| choice.compute),
            Ok(BrowserCompute::Cpu(CpuReason::Chosen))
        );
        assert_eq!(
            choose(&model(GPU_ONLY)?, &DeviceOrder::cpu_only(), F16).map(|choice| choice.compute),
            Err(CpuReason::Chosen)
        );
        Ok(())
    }

    #[test]
    fn the_provider_follows_the_compute() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            chosen(CPU_RUNNABLE, &DeviceOrder::automatic(), F16)?.provider(),
            Provider::WebGpu
        );
        assert_eq!(
            chosen(CPU_RUNNABLE, &DeviceOrder::cpu_only(), F16)?.provider(),
            Provider::Cpu
        );
        Ok(())
    }

    #[test]
    fn webgpu_takes_half_precision_only_where_the_adapter_has_shader_f16()
    -> Result<(), Box<dyn std::error::Error>> {
        let devices = DeviceOrder::automatic();
        assert_eq!(
            chosen(CPU_RUNNABLE, &devices, ShaderF16::Supported)?.variant,
            VariantKind::Fp16
        );
        assert_eq!(
            chosen(CPU_RUNNABLE, &devices, ShaderF16::Missing)?.variant,
            VariantKind::Fp32
        );
        assert_eq!(
            chosen(GPU_ONLY, &devices, ShaderF16::Missing)?.variant,
            VariantKind::Fp16
        );
        Ok(())
    }

    #[test]
    fn the_cpu_always_takes_full_precision() -> Result<(), Box<dyn std::error::Error>> {
        for shader_f16 in [ShaderF16::Supported, ShaderF16::Missing] {
            assert_eq!(
                chosen(CPU_RUNNABLE, &DeviceOrder::cpu_only(), shader_f16)?.variant,
                VariantKind::Fp32
            );
        }
        Ok(())
    }

    #[test]
    fn a_webgpu_failure_moves_a_cpu_capable_model_to_the_cpu_with_full_precision()
    -> Result<(), Box<dyn std::error::Error>> {
        let choice = chosen(CPU_RUNNABLE, &DeviceOrder::automatic(), F16)?;
        assert_eq!(choice.variant, VariantKind::Fp16);
        for (kind, reason) in [
            (FailureKind::WebGpuUnavailable, CpuReason::NoWebGpu),
            (FailureKind::ShaderF16Unavailable, CpuReason::NoShaderF16),
            (FailureKind::WebGpuFailed, CpuReason::WebGpuFailed),
        ] {
            assert_eq!(
                choice.after_failure(kind),
                GpuFailure::MoveToCpu(Choice {
                    compute: BrowserCompute::Cpu(reason),
                    variant: VariantKind::Fp32,
                    cpu_variant: Some(VariantKind::Fp32),
                }),
                "{kind:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn a_webgpu_failure_refuses_a_model_the_cpu_cannot_run()
    -> Result<(), Box<dyn std::error::Error>> {
        let choice = chosen(GPU_ONLY, &DeviceOrder::automatic(), F16)?;
        assert_eq!(
            choice.after_failure(FailureKind::WebGpuUnavailable),
            GpuFailure::CannotRun(CpuReason::NoWebGpu)
        );
        Ok(())
    }

    #[test]
    fn other_failures_and_runs_already_on_the_cpu_stay_as_they_are()
    -> Result<(), Box<dyn std::error::Error>> {
        let gpu = chosen(CPU_RUNNABLE, &DeviceOrder::automatic(), F16)?;
        let cpu = chosen(CPU_RUNNABLE, &DeviceOrder::cpu_only(), F16)?;
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

    #[test]
    fn the_shown_variant_is_the_one_a_run_would_use() -> Result<(), Box<dyn std::error::Error>> {
        let base = model(CPU_RUNNABLE)?;
        let kind =
            |devices: &DeviceOrder, shader_f16| shown_variant(&base, devices, shader_f16).variant;
        assert_eq!(kind(&DeviceOrder::automatic(), F16), VariantKind::Fp16);
        assert_eq!(
            kind(&DeviceOrder::automatic(), ShaderF16::Missing),
            VariantKind::Fp32
        );
        assert_eq!(kind(&DeviceOrder::cpu_only(), F16), VariantKind::Fp32);
        let turbo = model(GPU_ONLY)?;
        assert_eq!(
            shown_variant(&turbo, &DeviceOrder::cpu_only(), F16).variant,
            VariantKind::Fp16
        );
        Ok(())
    }
}
