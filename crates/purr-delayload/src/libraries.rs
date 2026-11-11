use std::path::PathBuf;

use crate::{cuda, error::DiscoveryError};

pub const VULKAN_LOADER: &str = "vulkan-1.dll";

/// Runtime libraries of each GPU backend that a binary must not link eagerly.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeLibraries {
    /// Dependencies come before their dependents, so probing in order names the library at fault.
    pub cuda: Vec<String>,
    pub vulkan: Vec<String>,
}

impl RuntimeLibraries {
    pub fn delay_loaded(&self) -> impl Iterator<Item = &str> {
        self.cuda.iter().chain(&self.vulkan).map(String::as_str)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cuda.is_empty() && self.vulkan.is_empty()
    }
}

/// What a build script knows about the binary being linked.
#[derive(Debug, Clone, Default)]
pub struct Request {
    /// Only the MSVC linker on Windows supports delay-loading.
    pub windows_msvc: bool,
    pub cuda: bool,
    pub vulkan: bool,
    pub cuda_path: Option<PathBuf>,
}

impl Request {
    #[must_use]
    pub fn from_cargo_env() -> Self {
        let cfg = |name: &str| std::env::var(name).ok();
        Self {
            windows_msvc: cfg("CARGO_CFG_TARGET_OS").as_deref() == Some("windows")
                && cfg("CARGO_CFG_TARGET_ENV").as_deref() == Some("msvc"),
            cuda: std::env::var_os("CARGO_FEATURE_CUDA").is_some(),
            vulkan: std::env::var_os("CARGO_FEATURE_VULKAN").is_some(),
            cuda_path: std::env::var_os("CUDA_PATH").map(PathBuf::from),
        }
    }

    pub fn resolve(&self) -> Result<RuntimeLibraries, DiscoveryError> {
        if !self.windows_msvc {
            return Ok(RuntimeLibraries::default());
        }
        let cuda = if self.cuda {
            let cuda_path = self
                .cuda_path
                .as_deref()
                .ok_or(DiscoveryError::CudaPathUnset)?;
            let dlls = cuda::locate(cuda_path)?;
            vec![dlls.cublas_lt, dlls.cublas]
        } else {
            Vec::new()
        };
        let vulkan = if self.vulkan {
            vec![VULKAN_LOADER.to_owned()]
        } else {
            Vec::new()
        };
        Ok(RuntimeLibraries { cuda, vulkan })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> Request {
        Request {
            windows_msvc: true,
            ..Request::default()
        }
    }

    #[test]
    fn other_targets_delay_load_nothing() {
        let request = Request {
            windows_msvc: false,
            cuda: true,
            vulkan: true,
            cuda_path: None,
        };
        assert_eq!(request.resolve(), Ok(RuntimeLibraries::default()));
    }

    #[test]
    fn no_gpu_feature_delay_loads_nothing() {
        assert_eq!(request().resolve(), Ok(RuntimeLibraries::default()));
    }

    #[test]
    fn vulkan_delay_loads_the_loader() {
        let request = Request {
            vulkan: true,
            ..request()
        };
        let libraries = request.resolve().unwrap();
        assert_eq!(libraries.vulkan, [VULKAN_LOADER]);
        assert_eq!(libraries.cuda, Vec::<String>::new());
    }

    #[test]
    fn cuda_without_a_toolkit_path_is_an_error() {
        let request = Request {
            cuda: true,
            ..request()
        };
        assert_eq!(request.resolve(), Err(DiscoveryError::CudaPathUnset));
    }

    #[test]
    fn cuda_delay_loads_both_cublas_dlls() -> std::io::Result<()> {
        let toolkit = tempfile::tempdir()?;
        let bin = toolkit.path().join("bin");
        std::fs::create_dir_all(&bin)?;
        std::fs::write(bin.join("cublas64_13.dll"), b"")?;
        std::fs::write(bin.join("cublasLt64_13.dll"), b"")?;
        let request = Request {
            cuda: true,
            cuda_path: Some(toolkit.path().to_path_buf()),
            ..request()
        };
        let libraries = request.resolve().unwrap();
        assert_eq!(libraries.cuda, ["cublasLt64_13.dll", "cublas64_13.dll"]);
        assert_eq!(
            libraries.delay_loaded().collect::<Vec<_>>(),
            ["cublasLt64_13.dll", "cublas64_13.dll"]
        );
        Ok(())
    }
}
