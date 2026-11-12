use std::collections::HashMap;

use crate::dev::{Device, list_devices};

use super::{
    backend::{AccelBackend, BuildStatus},
    runtime_library::detect_missing,
};

/// What the process knows about the acceleration it can use.
#[derive(Debug, Clone, Default)]
pub struct Inventory {
    pub devices: Vec<Device>,
    pub build: HashMap<AccelBackend, BuildStatus>,
    /// Backends that were built in but whose on-demand runtime library cannot be loaded.
    pub missing_runtime: HashMap<AccelBackend, String>,
}

impl Inventory {
    /// Enumerate the ggml devices registered in this process.
    #[must_use]
    pub fn detect() -> Self {
        let missing_runtime = detect_missing();
        Self {
            devices: list_devices(),
            build: AccelBackend::ALL
                .iter()
                .map(|backend| (*backend, backend.build_status()))
                .collect(),
            missing_runtime,
        }
    }

    #[must_use]
    pub fn build_status(&self, backend: AccelBackend) -> BuildStatus {
        self.build
            .get(&backend)
            .copied()
            .unwrap_or(BuildStatus::Unknown)
    }
}
