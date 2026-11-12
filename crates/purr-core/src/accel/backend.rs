use serde::{Deserialize, Serialize};

/// An acceleration backend a transcription can run on.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    strum::EnumString,
    strum::Display,
    strum::IntoStaticStr,
    strum::VariantArray,
    strum::VariantNames,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum AccelBackend {
    Cuda,
    #[strum(to_string = "hip", serialize = "rocm")]
    #[serde(alias = "rocm")]
    Hip,
    CoreMl,
    Metal,
    Sycl,
    Vulkan,
    Cpu,
}

/// Whether support for a backend was compiled into this binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildStatus {
    Built,
    NotBuilt,
    /// The build features of this crate cannot tell; only the runtime registry can.
    Unknown,
}

impl AccelBackend {
    pub const ALL: &'static [Self] = <Self as strum::VariantArray>::VARIANTS;

    /// Default priority, best first. Keep `Cpu` last: it is the last resort.
    pub const DEFAULT_PRIORITY: &'static [Self] = &[
        Self::Cuda,
        Self::Hip,
        Self::CoreMl,
        Self::Metal,
        Self::Sycl,
        Self::Vulkan,
        Self::Cpu,
    ];

    #[must_use]
    pub const fn pretty_name(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Hip => "ROCm",
            Self::CoreMl => "CoreML",
            Self::Metal => "Metal",
            Self::Sycl => "SYCL",
            Self::Vulkan => "Vulkan",
            Self::Cpu => "CPU",
        }
    }

    #[must_use]
    pub const fn build_status(self) -> BuildStatus {
        match self {
            Self::Cuda => feature_status(cfg!(feature = "cuda")),
            Self::CoreMl => feature_status(cfg!(feature = "coreml")),
            Self::Metal => feature_status(cfg!(feature = "metal")),
            Self::Vulkan => feature_status(cfg!(feature = "vulkan")),
            Self::Cpu => BuildStatus::Built,
            Self::Hip | Self::Sycl => BuildStatus::Unknown,
        }
    }

    /// What to do when the backend's runtime library cannot be loaded, where there is a usual fix.
    #[must_use]
    pub const fn runtime_library_hint(self) -> Option<&'static str> {
        match self {
            Self::Vulkan => Some(
                "install your GPU's Vulkan driver (on Linux, Intel and AMD use Mesa's Vulkan driver)",
            ),
            Self::Cuda | Self::Hip | Self::CoreMl | Self::Metal | Self::Sycl | Self::Cpu => None,
        }
    }

    /// Backend owning a ggml device, from the name of its ggml backend registry.
    #[must_use]
    pub fn from_ggml_registry_name(name: &str) -> Option<Self> {
        match name {
            "CUDA" => Some(Self::Cuda),
            "ROCm" => Some(Self::Hip),
            "Metal" => Some(Self::Metal),
            "SYCL" => Some(Self::Sycl),
            "Vulkan" => Some(Self::Vulkan),
            "CPU" => Some(Self::Cpu),
            _ => None,
        }
    }
}

const fn feature_status(enabled: bool) -> BuildStatus {
    if enabled {
        BuildStatus::Built
    } else {
        BuildStatus::NotBuilt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr as _;

    #[test]
    fn default_priority_lists_every_backend_once_and_ends_with_cpu() {
        let mut seen = Vec::new();
        for backend in AccelBackend::DEFAULT_PRIORITY {
            assert!(!seen.contains(backend));
            seen.push(*backend);
        }
        for backend in AccelBackend::ALL {
            assert!(seen.contains(backend));
        }
        assert_eq!(
            AccelBackend::DEFAULT_PRIORITY.last(),
            Some(&AccelBackend::Cpu)
        );
    }

    #[test]
    fn rocm_is_an_alias_of_hip() {
        assert_eq!(AccelBackend::from_str("rocm"), Ok(AccelBackend::Hip));
        assert_eq!(AccelBackend::from_str("hip"), Ok(AccelBackend::Hip));
    }

    #[test]
    fn serde_names_match_parse_names() {
        for backend in AccelBackend::ALL {
            let json = serde_json::to_string(backend).unwrap();
            let name: String = serde_json::from_str(&json).unwrap();
            assert_eq!(AccelBackend::from_str(&name), Ok(*backend));
            assert_eq!(
                serde_json::from_str::<AccelBackend>(&json).unwrap(),
                *backend
            );
        }
    }
}
