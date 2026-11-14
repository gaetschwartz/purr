use crate::hooks::use_models::GpuSupport;
use purr_common::{
    browser_compute::{BrowserCompute, CpuReason},
    devices::DeviceOrder,
};

/// Where the next browser run is expected to go: what the settings ask for, unless the browser's
/// adapter already rules WebGPU out. The run itself still moves to the CPU if WebGPU fails.
#[must_use]
pub fn expected_browser_compute(
    devices: &DeviceOrder,
    gpu: &GpuSupport,
    needs_shader_f16: bool,
) -> BrowserCompute {
    match (BrowserCompute::requested(devices), gpu) {
        (cpu @ BrowserCompute::Cpu(_), _) => cpu,
        (BrowserCompute::WebGpu, GpuSupport::Unavailable) => {
            BrowserCompute::Cpu(CpuReason::NoWebGpu)
        }
        (
            BrowserCompute::WebGpu,
            GpuSupport::Available {
                shader_f16: false, ..
            },
        ) if needs_shader_f16 => BrowserCompute::Cpu(CpuReason::NoShaderF16),
        (BrowserCompute::WebGpu, GpuSupport::Checking | GpuSupport::Available { .. }) => {
            BrowserCompute::WebGpu
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(shader_f16: bool) -> GpuSupport {
        GpuSupport::Available {
            name: None,
            shader_f16,
        }
    }

    #[test]
    fn the_settings_decide_while_the_adapter_is_still_being_looked_for() {
        assert_eq!(
            expected_browser_compute(&DeviceOrder::automatic(), &GpuSupport::Checking, true),
            BrowserCompute::WebGpu
        );
        assert_eq!(
            expected_browser_compute(&DeviceOrder::cpu_only(), &GpuSupport::Checking, false),
            BrowserCompute::Cpu(CpuReason::Chosen)
        );
    }

    #[test]
    fn choosing_the_cpu_wins_over_a_capable_adapter() {
        assert_eq!(
            expected_browser_compute(&DeviceOrder::cpu_only(), &adapter(true), false),
            BrowserCompute::Cpu(CpuReason::Chosen)
        );
    }

    #[test]
    fn a_browser_without_an_adapter_runs_on_the_cpu() {
        assert_eq!(
            expected_browser_compute(&DeviceOrder::automatic(), &GpuSupport::Unavailable, false),
            BrowserCompute::Cpu(CpuReason::NoWebGpu)
        );
    }

    #[test]
    fn a_missing_shader_f16_only_matters_to_models_that_need_it() {
        assert_eq!(
            expected_browser_compute(&DeviceOrder::automatic(), &adapter(false), true),
            BrowserCompute::Cpu(CpuReason::NoShaderF16)
        );
        assert_eq!(
            expected_browser_compute(&DeviceOrder::automatic(), &adapter(false), false),
            BrowserCompute::WebGpu
        );
        assert_eq!(
            expected_browser_compute(&DeviceOrder::automatic(), &adapter(true), true),
            BrowserCompute::WebGpu
        );
    }
}
