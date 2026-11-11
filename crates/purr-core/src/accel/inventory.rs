use std::collections::HashMap;

use crate::dev::{Device, list_devices};

use super::backend::{AccelBackend, BuildStatus};

/// What the process knows about the acceleration it can use.
#[derive(Debug, Clone, Default)]
pub struct Inventory {
    pub devices: Vec<Device>,
    pub build: HashMap<AccelBackend, BuildStatus>,
}

impl Inventory {
    /// Enumerate the ggml devices registered in this process.
    #[must_use]
    pub fn detect() -> Self {
        Self {
            devices: list_devices(),
            build: AccelBackend::ALL
                .iter()
                .map(|backend| (*backend, backend.build_status()))
                .collect(),
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
