use crate::hooks::use_models::GpuSupport;
use purr_common::{
    browser_engine::{BrowserEngine, CpuReason},
    devices::DeviceOrder,
};

/// Where the next browser run is expected to go: what the settings ask for, unless the browser's
/// adapter already rules WebGPU out. The run itself still moves to the CPU if WebGPU fails.
#[must_use]
pub fn expected_browser_engine(
    devices: &DeviceOrder,
    gpu: &GpuSupport,
    needs_shader_f16: bool,
) -> BrowserEngine {
    match (BrowserEngine::requested(devices), gpu) {
        (cpu @ BrowserEngine::Cpu(_), _) => cpu,
        (BrowserEngine::WebGpu, GpuSupport::Unavailable) => BrowserEngine::Cpu(CpuReason::NoWebGpu),
        (
            BrowserEngine::WebGpu,
            GpuSupport::Available {
                shader_f16: false, ..
            },
        ) if needs_shader_f16 => BrowserEngine::Cpu(CpuReason::NoShaderF16),
        (BrowserEngine::WebGpu, GpuSupport::Checking | GpuSupport::Available { .. }) => {
            BrowserEngine::WebGpu
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
            expected_browser_engine(&DeviceOrder::automatic(), &GpuSupport::Checking, true),
            BrowserEngine::WebGpu
        );
        assert_eq!(
            expected_browser_engine(&DeviceOrder::cpu_only(), &GpuSupport::Checking, false),
            BrowserEngine::Cpu(CpuReason::Chosen)
        );
    }

    #[test]
    fn choosing_the_cpu_wins_over_a_capable_adapter() {
        assert_eq!(
            expected_browser_engine(&DeviceOrder::cpu_only(), &adapter(true), false),
            BrowserEngine::Cpu(CpuReason::Chosen)
        );
    }

    #[test]
    fn a_browser_without_an_adapter_runs_on_the_cpu() {
        assert_eq!(
            expected_browser_engine(&DeviceOrder::automatic(), &GpuSupport::Unavailable, false),
            BrowserEngine::Cpu(CpuReason::NoWebGpu)
        );
    }

    #[test]
    fn a_missing_shader_f16_only_matters_to_models_that_need_it() {
        assert_eq!(
            expected_browser_engine(&DeviceOrder::automatic(), &adapter(false), true),
            BrowserEngine::Cpu(CpuReason::NoShaderF16)
        );
        assert_eq!(
            expected_browser_engine(&DeviceOrder::automatic(), &adapter(false), false),
            BrowserEngine::WebGpu
        );
        assert_eq!(
            expected_browser_engine(&DeviceOrder::automatic(), &adapter(true), true),
            BrowserEngine::WebGpu
        );
    }
}
