use purr_core::accel::{
    backend::{AccelBackend, BuildStatus},
    build_info::BuildInfo,
};

/// The backends this build asks transcribe.cpp for: the CPU only. Its CPU code is one module per instruction
/// set tier, chosen at run time, so no single set of CPU features is compiled in.
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

const fn backend_status(backend: AccelBackend) -> BuildStatus {
    match backend {
        AccelBackend::Cpu => BuildStatus::Built,
        AccelBackend::Cuda
        | AccelBackend::CoreMl
        | AccelBackend::Hip
        | AccelBackend::Metal
        | AccelBackend::Sycl
        | AccelBackend::Vulkan => BuildStatus::NotBuilt,
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
    fn the_cpu_is_built_and_cuda_is_not() {
        let info = build_info();
        assert_eq!(info.status(AccelBackend::Cpu), BuildStatus::Built);
        assert_eq!(info.status(AccelBackend::Cuda), BuildStatus::NotBuilt);
    }

    #[test]
    fn no_cpu_feature_is_claimed() {
        assert_eq!(build_info().cpu, []);
    }
}
