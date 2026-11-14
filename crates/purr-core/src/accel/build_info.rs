use std::collections::HashMap;

use purr_common::accel::CpuFeature;

use super::backend::{AccelBackend, BuildStatus};

/// What one engine was built with: the backends it can use and the instruction sets its CPU
/// code was compiled for.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BuildInfo {
    pub backends: HashMap<AccelBackend, BuildStatus>,
    pub cpu: Vec<CpuFeature>,
}

impl BuildInfo {
    /// Whether support for `backend` was compiled in; `Unknown` for a backend the build does not
    /// speak for.
    #[must_use]
    pub fn status(&self, backend: AccelBackend) -> BuildStatus {
        self.backends
            .get(&backend)
            .copied()
            .unwrap_or(BuildStatus::Unknown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_backend_the_build_does_not_list_is_unknown() {
        let build = BuildInfo {
            backends: HashMap::from([
                (AccelBackend::Cpu, BuildStatus::Built),
                (AccelBackend::Cuda, BuildStatus::NotBuilt),
            ]),
            cpu: vec![CpuFeature::Avx2],
        };
        assert_eq!(build.status(AccelBackend::Cpu), BuildStatus::Built);
        assert_eq!(build.status(AccelBackend::Cuda), BuildStatus::NotBuilt);
        assert_eq!(build.status(AccelBackend::Hip), BuildStatus::Unknown);
    }
}
