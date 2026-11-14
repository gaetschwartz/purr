use std::{collections::HashMap, path::PathBuf};

use crate::{dev::Device, engine::Engine};

use super::{
    backend::{AccelBackend, BuildStatus},
    build_info::BuildInfo,
    runtime_dir::{Environment, search_dirs},
    runtime_library::detect_missing,
};

/// What the process knows about the acceleration it can use.
#[derive(Debug, Clone, Default)]
pub struct Inventory {
    pub devices: Vec<Device>,
    pub build: BuildInfo,
    /// Backends that were built in but whose on-demand runtime library cannot be loaded.
    pub missing_runtime: HashMap<AccelBackend, String>,
    /// The directories searched for runtime libraries before the system's own search.
    pub runtime_dirs: Vec<PathBuf>,
}

impl Inventory {
    /// What `engine` enumerates and was built with, with the runtime libraries it is missing.
    ///
    /// The runtime libraries are opened before the engine enumerates its devices, so the
    /// delay-loaded imports its backends make while registering bind to them.
    #[must_use]
    pub fn detect(engine: &dyn Engine) -> Self {
        let dirs = search_dirs(&Environment::current());
        let missing_runtime = detect_missing(&dirs);
        Self {
            devices: engine.devices(),
            build: engine.build_info(),
            missing_runtime,
            runtime_dirs: dirs.into_iter().map(|dir| dir.path).collect(),
        }
    }

    #[must_use]
    pub fn build_status(&self, backend: AccelBackend) -> BuildStatus {
        self.build.status(backend)
    }
}
