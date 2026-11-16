use purr_core::accel::{
    backend::{AccelBackend, BuildStatus},
    build_info::BuildInfo,
};

/// The backends this build asks transcribe.cpp for: the CPU, and the GPU backends of the cargo
/// features that are on. Its CPU code is one module per instruction set tier, chosen at run time,
/// so no single set of CPU features is compiled in.
#[must_use]
pub fn build_info() -> BuildInfo {
    BuildInfo {
        backends: AccelBackend::ALL
            .iter()
            .map(|backend| (*backend, backend_status(*backend)))
            .collect(),
        cpu: Vec::new(),
    }
}

const fn built(backend: AccelBackend) -> bool {
    match backend {
        AccelBackend::Cpu => true,
        AccelBackend::Cuda => cfg!(feature = "cuda"),
        AccelBackend::Metal => cfg!(all(feature = "metal", target_vendor = "apple")),
        AccelBackend::Vulkan => cfg!(feature = "vulkan"),
        AccelBackend::CoreMl | AccelBackend::Hip | AccelBackend::Sycl => false,
    }
}

const fn backend_status(backend: AccelBackend) -> BuildStatus {
    if built(backend) {
        BuildStatus::Built
    } else {
        BuildStatus::NotBuilt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_backend_has_a_build_status() {
        assert_eq!(build_info().backends.len(), AccelBackend::ALL.len());
    }

    #[test]
    fn the_cpu_is_always_built() {
        assert_eq!(build_info().status(AccelBackend::Cpu), BuildStatus::Built);
    }

    #[test]
    fn a_gpu_backend_is_built_when_its_feature_is_on() {
        let info = build_info();
        let status = |on: bool| {
            if on {
                BuildStatus::Built
            } else {
                BuildStatus::NotBuilt
            }
        };
        assert_eq!(
            info.status(AccelBackend::Cuda),
            status(cfg!(feature = "cuda"))
        );
        assert_eq!(
            info.status(AccelBackend::Vulkan),
            status(cfg!(feature = "vulkan"))
        );
    }

    #[test]
    fn no_cpu_feature_is_claimed() {
        assert_eq!(build_info().cpu, []);
    }
}
