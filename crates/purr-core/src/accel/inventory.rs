use std::{collections::HashMap, path::PathBuf};

use crate::dev::{Device, list_devices};

use super::{
    backend::{AccelBackend, BuildStatus},
    runtime_dir::{Environment, search_dirs},
    runtime_library::detect_missing,
};

/// What the process knows about the acceleration it can use.
#[derive(Debug, Clone, Default)]
pub struct Inventory {
    pub devices: Vec<Device>,
    pub build: HashMap<AccelBackend, BuildStatus>,
    /// Backends that were built in but whose on-demand runtime library cannot be loaded.
    pub missing_runtime: HashMap<AccelBackend, String>,
    /// The directories searched for runtime libraries before the system's own search.
    pub runtime_dirs: Vec<PathBuf>,
}

impl Inventory {
    /// Enumerate the ggml devices registered in this process.
    #[must_use]
    pub fn detect() -> Self {
        let dirs = search_dirs(&Environment::current());
        let missing_runtime = detect_missing(&dirs);
        Self {
            devices: list_devices(),
            build: AccelBackend::ALL
                .iter()
                .map(|backend| (*backend, backend.build_status()))
                .collect(),
            missing_runtime,
            runtime_dirs: dirs.into_iter().map(|dir| dir.path).collect(),
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
