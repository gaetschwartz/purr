use serde::{Deserialize, Serialize};

use super::backend::AccelBackend;

/// How the acceleration backend is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccelMode {
    /// Try the priority list from best to worst, moving to the next backend when one is
    /// unavailable or fails.
    Auto,
    /// Use `backend`. Without `allow_fallback` there is no fallback: an unavailable or failing
    /// backend is an error. With it, the priority list (ending in CPU) is tried after `backend`.
    Force {
        backend: AccelBackend,
        allow_fallback: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccelConfig {
    pub mode: AccelMode,
    /// Backends tried in `Auto` mode and after a forced backend with fallback, best first.
    /// Duplicates are ignored and `Cpu` is appended when absent. Backends listed after `Cpu`
    /// are never reached since the CPU always works.
    pub priority: Vec<AccelBackend>,
}

impl Default for AccelConfig {
    fn default() -> Self {
        Self {
            mode: AccelMode::Auto,
            priority: AccelBackend::DEFAULT_PRIORITY.to_vec(),
        }
    }
}

impl AccelConfig {
    #[must_use]
    pub fn cpu_only() -> Self {
        Self::forced(AccelBackend::Cpu, false)
    }

    #[must_use]
    pub fn forced(backend: AccelBackend, allow_fallback: bool) -> Self {
        Self {
            mode: AccelMode::Force {
                backend,
                allow_fallback,
            },
            ..Self::default()
        }
    }

    #[must_use]
    pub fn with_priority(mut self, priority: Vec<AccelBackend>) -> Self {
        self.priority = priority;
        self
    }

    /// Backends to attempt, in order.
    #[must_use]
    pub fn attempt_order(&self) -> Vec<AccelBackend> {
        let mut order = Vec::new();
        match self.mode {
            AccelMode::Auto => {}
            AccelMode::Force {
                backend,
                allow_fallback: false,
            } => return vec![backend],
            AccelMode::Force {
                backend,
                allow_fallback: true,
            } => order.push(backend),
        }
        for backend in self.priority.iter().chain([&AccelBackend::Cpu]) {
            if !order.contains(backend) {
                order.push(*backend);
            }
        }
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use AccelBackend::{Cpu, Cuda, Metal, Vulkan};

    #[test]
    fn auto_follows_the_default_priority() {
        assert_eq!(
            AccelConfig::default().attempt_order(),
            AccelBackend::DEFAULT_PRIORITY
        );
    }

    #[test]
    fn auto_always_ends_with_cpu() {
        let config = AccelConfig::default().with_priority(vec![Vulkan, Cuda]);
        assert_eq!(config.attempt_order(), [Vulkan, Cuda, Cpu]);
    }

    #[test]
    fn auto_ignores_duplicates() {
        let config = AccelConfig::default().with_priority(vec![Cuda, Vulkan, Cuda, Cpu, Vulkan]);
        assert_eq!(config.attempt_order(), [Cuda, Vulkan, Cpu]);
    }

    #[test]
    fn empty_priority_is_cpu_only() {
        let config = AccelConfig::default().with_priority(Vec::new());
        assert_eq!(config.attempt_order(), [Cpu]);
    }

    #[test]
    fn forced_without_fallback_tries_only_that_backend() {
        assert_eq!(AccelConfig::forced(Vulkan, false).attempt_order(), [Vulkan]);
    }

    #[test]
    fn forced_with_fallback_tries_it_first_then_the_priority_list() {
        let config = AccelConfig::forced(Vulkan, true).with_priority(vec![Cuda, Metal, Vulkan]);
        assert_eq!(config.attempt_order(), [Vulkan, Cuda, Metal, Cpu]);
    }

    #[test]
    fn cpu_only_never_falls_back() {
        assert_eq!(AccelConfig::cpu_only().attempt_order(), [Cpu]);
    }
}
